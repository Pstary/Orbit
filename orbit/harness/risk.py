"""Bash 命令四级风险分级器。

分级管线（不做简单的命令名黑名单）：

    命令解析 → 组合命令拆分 → 命令替换/嵌套 shell 递归 → 权限提升检测
    → 特殊命令分析（rm / chmod / dd / del / rd / Remove-Item …）
    → 敏感路径检测 → 重定向/副作用检测 → 命令名分级 → 未知命令兜底

四级风险与处置动作：

    LOW       本地直接执行      只读/导航类命令（pwd、ls、cat、git status …）
    MEDIUM    Docker 沙箱执行   可能产生副作用（跑代码、包管理、网络、文件改动…）
    HIGH      必须人工审批      系统状态修改（删除、权限、进程、服务…）或读取系统敏感路径
    CRITICAL  直接拒绝          不可逆破坏（rm -rf /、mkfs、写裸盘、关机、fork 炸弹…）

设计要点：
- 组合命令（&&/||/;/|/&）逐段分级取最高级，`ls && rm -rf /` 不会被第一段 `ls` 伪装成低风险。
- 命令替换 `$(...)`、反引号、`bash -c "..."`、`powershell -Command "..."` 内部脚本递归分级。
- 跨平台：同时覆盖 Unix 与 Windows（del/rd/format/diskpart/Remove-Item/icacls/taskkill…）。
- 返回 RiskResult(level, reasons)：reasons 列出命中的每条规则，便于审计日志与 trace。
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


class RiskLevel(IntEnum):
    """Bash 命令风险等级（数值越大风险越高，组合命令取最高级）。"""

    LOW = 0       # 本地直接执行
    MEDIUM = 1    # Docker 临时沙箱执行
    HIGH = 2      # 必须人工审批
    CRITICAL = 3  # 直接拒绝


# 风险等级对应的处置动作。
RISK_ACTIONS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "local_execution",
    RiskLevel.MEDIUM: "docker_sandbox",
    RiskLevel.HIGH: "human_approval",
    RiskLevel.CRITICAL: "reject",
}


@dataclass
class RiskResult:
    """分级结果：等级 + 命中规则说明（审计/trace 直接可用）。"""

    level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    command: str = ""

    @property
    def level_name(self) -> str:
        return self.level.name

    @property
    def action(self) -> str:
        return RISK_ACTIONS[self.level]

    def to_dict(self) -> dict:
        return {
            "risk_level": self.level.name,
            "risk_action": self.action,
            "risk_reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# 命令名分级表（跨平台；basename 小写比较，自动去掉 .exe 后缀）
# ---------------------------------------------------------------------------

# LOW：只读 / 导航类命令，不修改系统状态。
LOW_COMMANDS = {
    # 导航 / 目录列举
    "pwd", "ls", "ll", "la", "dir", "cd", "pushd", "popd", "tree",
    # 文件查看
    "cat", "head", "tail", "less", "more", "type", "tac", "nl", "strings",
    "xxd", "hexdump", "od",
    # 检索 / 文本处理（纯读取）
    "grep", "egrep", "fgrep", "rg", "find", "wc", "sort", "uniq", "cut",
    "awk", "diff", "cmp", "comm", "rev",
    # 系统信息（只读）
    "date", "cal", "uptime", "whoami", "id", "groups", "who", "w", "last",
    "hostname", "uname", "ver", "systeminfo", "sw_vers",
    "which", "whereis", "where", "type", "file", "stat", "basename", "dirname",
    "realpath", "readlink",
    "du", "df", "ps", "tasklist", "free", "vmstat", "iostat", "lsof",
    "dmesg", "printenv", "env", "getent",
    # 校验和
    "md5sum", "sha1sum", "sha256sum", "shasum", "cksum",
    # 输出（无重定向时只读）
    "echo",
    # 网络查看（只读；配置变更用 ip/route/ipconfig /release 等）
    "ipconfig",
}

# MEDIUM：可能产生副作用——执行任意代码、包管理、网络访问、文件增改。
MEDIUM_COMMANDS = {
    # 运行代码（可执行任意逻辑）
    "python", "python3", "py", "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php", "java", "javac", "dotnet", "mvn", "gradle",
    "go", "rustc", "cargo", "make", "cmake", "ninja", "gcc", "g++",
    "clang", "clang++",
    # 测试运行器（执行项目代码）
    "pytest", "tox", "nox", "coverage",
    # 包管理
    "pip", "pip3", "pipx", "conda", "uv", "poetry", "pipenv", "virtualenv",
    "npm", "npx", "yarn", "pnpm", "gem", "bundle", "rake",
    "winget", "choco", "nuget", "msiexec",
    # 网络（下载/上传/请求外部资源）
    "curl", "wget",
    # git：有副作用（clone/pull/push/commit…），只读子命令在分析时降为 LOW
    "git",
    # 压缩 / 解压
    "tar", "gzip", "gunzip", "bzip2", "xz", "zip", "unzip", "7z",
    # 嵌套 shell / 脚本宿主（-c/-Command 内部脚本会递归分级，兜底 MEDIUM）
    "bash", "sh", "zsh", "fish", "ksh", "dash",
    "cmd", "powershell", "pwsh", "cscript", "wscript",
    "start", "call",
    # Windows 文件属性 / 证书工具（可下载/解码文件）
    "attrib", "certutil",
    # 容器（daemon 权限等同 root，但常见用法是构建/跑测试，先 MEDIUM）
    "docker", "podman",
}

# HIGH：修改系统状态——删除、文件增改、权限、进程、服务、用户、包安装、网络/挂载配置。
HIGH_COMMANDS = {
    # 删除
    "rm", "rmdir", "del", "erase", "rd",
    # 文件增改（创建/复制/移动/链接/打补丁）——工作区变更，走人工审批门
    "touch", "mkdir", "md", "cp", "copy", "xcopy", "robocopy",
    "mv", "move", "ln", "install", "patch",
    # 权限
    "chmod", "chown", "chgrp", "icacls", "cacls", "takeown",
    # 进程
    "kill", "killall", "pkill", "taskkill", "tskill",
    # 服务
    "systemctl", "service", "launchctl", "sc", "schtasks",
    # 用户 / 用户组 / 密码
    "useradd", "userdel", "usermod", "groupadd", "groupdel", "groupmod",
    "passwd",
    # 网络 / 防火墙配置
    "iptables", "ip6tables", "nft", "ip", "route", "netsh",
    # 挂载
    "mount", "umount",
    # 系统级包安装
    "apt", "apt-get", "aptitude", "yum", "dnf", "pacman", "zypper", "brew",
    # 内核 / 调度
    "sysctl",
    # 裸写
    "dd",
    # 计划任务
    "crontab", "at",
    # PowerShell 删除 cmdlet
    "remove-item",
}

# CRITICAL：命令名本身即直接拒绝（不可逆 / 系统级破坏）。
CRITICAL_COMMANDS = {
    "mkfs", "format", "diskpart", "bcdedit",
    "shutdown", "reboot", "halt", "poweroff",
}

# 权限提升：即使后面跟的是低危命令，执行上下文已变化（sudo ls → HIGH）。
PRIVILEGE_COMMANDS = {"sudo", "su", "doas", "runas"}

# 嵌套 shell：这些宿主的 -c/-Command 后面跟的是一段完整脚本，需要递归分级。
SHELL_HOST_COMMANDS = {
    "bash", "sh", "zsh", "fish", "ksh", "dash",
    "cmd", "powershell", "pwsh",
}
_SHELL_SCRIPT_FLAGS = {"-c", "-lc", "-cl", "-command", "--command", "/c", "/k", "-commandtext"}
_SHELL_ENCODED_FLAGS = {"-enc", "-encodedcommand", "-e"}  # PowerShell base64，无法静态检查

# 只读 git 子命令（与 permissions.py 中既有口径保持一致）。
_READ_ONLY_GIT_WORDS = {"branch", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}

# Windows net 子命令：服务/用户/共享变更 → HIGH。
_HIGH_NET_WORDS = {"start", "stop", "user", "localgroup", "group", "share", "accounts", "use"}
# Windows reg 子命令：注册表写操作 → HIGH；读操作 → MEDIUM。
_HIGH_REG_WORDS = {"add", "delete", "import", "save", "load", "restore", "copy"}

# 重定向操作符（写文件副作用）。
_REDIRECT_OPS = {">", ">>", ">|", "&>", "&>>", "1>", "1>>", "2>", "2>>"}
# 重定向到这些设备是安全/惯用法，不算敏感目标。
_SAFE_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "nul"}

# 整命令级别的 CRITICAL 正则（跨段、纯语法形态，逐段分析难以覆盖）。
_CRITICAL_PATTERNS = [
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\b(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+|doas\s+)?(bash|sh|zsh|fish|dash)\b",
     "pipe network download into shell"),
]

# 组合命令分隔符（&&/||/;/|/&/换行）。
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||\||;|&|\r?\n")

_MAX_REASONS = 12
_MAX_RECURSION_DEPTH = 3


# ---------------------------------------------------------------------------
# 敏感路径（跨平台；同时按 / 和 \ 两种分隔符匹配）
# ---------------------------------------------------------------------------

def _normalize_separators(path: str) -> str:
    """统一成分隔符 / 并小写，便于跨平台比较（Windows 路径大小写不敏感）。

    根目录必须原样保留：rstrip("/") 会把 "/" 清空、把 "C:/" 变成 "c:"，
    导致 `rm -rf /`、`rd /s C:\\` 这类针对根目录的命令漏判。
    """
    normalized = path.replace("\\", "/").lower()
    if normalized == "/":
        return "/"
    drive_root = re.match(r"^([a-z]):/*$", normalized)
    if drive_root:
        return f"{drive_root.group(1)}:/"
    return normalized.rstrip("/")


def _system_drive() -> str:
    drive = os.environ.get("SystemDrive") or "C:"
    return drive.rstrip(":/\\") + ":\\"


def _home_root() -> str:
    try:
        return str(Path.home())
    except RuntimeError:
        return ""


def _build_critical_path_prefixes() -> set[str]:
    """前缀匹配：命中目录本身或其下任意路径都算敏感。"""
    prefixes = {
        # Unix 系统目录
        "/boot", "/bin", "/sbin", "/usr", "/etc", "/root",
        "/dev", "/proc", "/sys", "/lib", "/lib64",
    }
    drive = _system_drive()
    prefixes.update({
        f"{drive}Windows",
        f"{drive}Program Files",
        f"{drive}Program Files (x86)",
        f"{drive}ProgramData",
        f"{drive}Boot",
        f"{drive}$Recycle.Bin",
        f"{drive}System Volume Information",
    })
    return {_normalize_separators(p) for p in prefixes}


def _build_critical_path_exact() -> set[str]:
    """精确匹配：根目录 / 用户目录本身（不能前缀匹配，否则工作区路径全部误伤）。"""
    exact = {"/", "/home"}
    home = _home_root()
    if home:
        exact.add(home)
        # C:\Users 本身（C:\Users\<name>\工作区 不命中）
        parent = os.path.dirname(home)
        if parent:
            exact.add(parent)
    drive_root = _system_drive()
    exact.add(drive_root.rstrip("\\"))
    exact.add(drive_root)
    return {_normalize_separators(p) for p in exact}


_CRITICAL_PATH_PREFIXES = _build_critical_path_prefixes()
_CRITICAL_PATH_EXACT = _build_critical_path_exact()


def _expand_path(token: str) -> str:
    """展开 ~、$VAR、%VAR% 后做规范化（命令字符串尚未经过 shell 展开）。"""
    try:
        expanded = os.path.expandvars(os.path.expanduser(token))
        return os.path.normpath(expanded)
    except Exception:  # noqa: BLE001
        return token


def is_critical_path(path: str) -> bool:
    """目标路径是否为系统敏感路径（删除/写入 → CRITICAL，读取 → HIGH）。"""
    if not path:
        return False
    normalized = _normalize_separators(_expand_path(path))
    if not normalized:
        return False
    if normalized in _CRITICAL_PATH_EXACT:
        return True
    for prefix in _CRITICAL_PATH_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return True
    # 裸设备：/dev/sda、/dev/nvme0n1、\\.\PhysicalDrive0
    if re.match(r"^/dev/(sd|nvme|disk|vd|xvd|hd)[a-z0-9]", normalized):
        return True
    if "physicaldrive" in normalized or normalized.startswith("//./"):
        return True
    return False


# ---------------------------------------------------------------------------
# 分级入口
# ---------------------------------------------------------------------------

def classify_command(command: str) -> RiskResult:
    """对一条完整 shell 命令做四级风险分级。"""
    if not command or not command.strip():
        return RiskResult(RiskLevel.LOW, ["empty command"], command or "")

    command = command.strip()

    # 第一层：整命令级别的明确危险形态（fork 炸弹、curl|sh 等跨段模式）。
    for pattern, reason in _CRITICAL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return RiskResult(RiskLevel.CRITICAL, [f"critical pattern: {reason}"], command)

    # 第二层：拆组合命令，逐段分级，整体取最高级。
    segments = _split_segments(command)
    overall = RiskResult(RiskLevel.LOW, [], command)
    for index, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue
        result = _classify_segment(segment, depth=0)
        prefix = f"segment {index + 1}: " if len(segments) > 1 else ""
        if result.level > overall.level:
            overall.level = result.level
            overall.reasons = [prefix + r for r in result.reasons]
        elif result.level == overall.level:
            overall.reasons.extend(prefix + r for r in result.reasons)

    if not overall.reasons:
        overall.reasons = ["no risk rules matched"]
    # 去重保序，避免 reasons 无限膨胀。
    overall.reasons = list(dict.fromkeys(overall.reasons))[:_MAX_REASONS]
    return overall


def _split_segments(command: str) -> list[str]:
    """在引号外按 &&/||/;/|/&/换行 拆分组合命令。"""
    segments: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        # 双字符操作符优先。
        two = command[i:i + 2]
        if two in ("&&", "||"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "&") or command[i:i + 1] == "\n":
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return segments


def _classify_segment(segment: str, *, depth: int) -> RiskResult:
    """对单个命令段分级（不含 &&/||/; 等组合操作符）。"""
    if depth > _MAX_RECURSION_DEPTH:
        return RiskResult(RiskLevel.MEDIUM, ["deeply nested shell, default to sandbox"])

    segment = segment.strip()
    if not segment:
        return RiskResult(RiskLevel.LOW, ["empty segment"])

    level = RiskLevel.LOW
    reasons: list[str] = []

    def bump(new_level: RiskLevel, reason: str) -> None:
        nonlocal level
        if new_level > level:
            level = new_level
        reasons.append(reason)

    # 第三层：命令替换 $(...) / 反引号 —— 内部内容会真实执行，递归分级。
    for inner in _extract_substitutions(segment):
        inner_result = classify_command(inner)
        if inner_result.level >= RiskLevel.MEDIUM:
            bump(
                max(RiskLevel.MEDIUM, inner_result.level),
                f"command substitution runs: {'; '.join(inner_result.reasons[:3])}",
            )

    # 解析 token（posix=False 保留 Windows 反斜杠路径）。
    try:
        tokens = [_strip_quotes(t) for t in shlex.split(segment, posix=False) if t]
    except ValueError as exc:
        return RiskResult(RiskLevel.HIGH, [f"invalid shell syntax: {exc}"])
    if not tokens:
        return RiskResult(RiskLevel.LOW, ["empty segment"])

    # 第四层：嵌套 shell（bash -c / cmd /c / powershell -Command …）递归分级脚本内容。
    nested = _extract_nested_script(tokens)
    if nested is not None:
        script, encoded = nested
        if encoded:
            bump(RiskLevel.HIGH, "encoded shell command (-enc/-EncodedCommand) cannot be inspected")
        elif script:
            inner_result = classify_command(script)
            bump(
                max(RiskLevel.MEDIUM, inner_result.level),
                f"nested shell script: {'; '.join(inner_result.reasons[:3])}",
            )
        else:
            bump(RiskLevel.MEDIUM, "shell host invocation")

    # 第五层：权限提升（sudo/su/doas/runas）。
    for token in tokens:
        if _base_command_name(token) in PRIVILEGE_COMMANDS:
            bump(RiskLevel.HIGH, "privilege escalation (sudo/su/doas/runas)")
            break

    # 第六层：重定向目标（> / >> / 2> / tee）——写文件副作用，敏感目标直接 CRITICAL。
    redirect_targets = _extract_redirect_targets(tokens)
    for target in redirect_targets:
        low = target.lower()
        if low in _SAFE_REDIRECT_TARGETS:
            continue
        if is_critical_path(target):
            bump(RiskLevel.CRITICAL, f"redirect writes to sensitive system path: {target}")
        else:
            bump(RiskLevel.MEDIUM, "output redirect writes to a file")

    name = _base_command_name(tokens[0])
    args = tokens[1:]

    # 第七层：特殊命令专门分析（命令 + 参数 + 目标路径联合判断）。
    special = _analyze_special_command(name, args, tokens)
    if special is not None:
        bump(special.level, "; ".join(special.reasons))
    elif name in CRITICAL_COMMANDS or any(name.startswith(c) for c in ("mkfs",)):
        bump(RiskLevel.CRITICAL, f"critical command: {name}")
    elif name in PRIVILEGE_COMMANDS:
        bump(RiskLevel.HIGH, f"privilege escalation command: {name}")
    elif name in HIGH_COMMANDS:
        bump(RiskLevel.HIGH, f"high-risk command: {name}")
    elif name == "git":
        # git 只读子命令降为 LOW，其余按 MEDIUM。
        if args and _first_word(args) in _READ_ONLY_GIT_WORDS and not _has_redirect(tokens):
            bump(RiskLevel.LOW, f"read-only git subcommand: git {_first_word(args)}")
        else:
            bump(RiskLevel.MEDIUM, "git command with possible side effects")
    elif name == "net":
        word = _first_word(args)
        if word in _HIGH_NET_WORDS:
            bump(RiskLevel.HIGH, f"windows net {word} modifies services/users/shares")
        else:
            bump(RiskLevel.MEDIUM, "windows net command")
    elif name == "reg":
        word = _first_word(args)
        if word in _HIGH_REG_WORDS:
            bump(RiskLevel.HIGH, f"windows registry write: reg {word}")
        else:
            bump(RiskLevel.MEDIUM, "windows registry access")
    elif name == "ipconfig":
        if any(a.lower() in ("/release", "/renew") for a in args):
            bump(RiskLevel.HIGH, "ipconfig /release|/renew changes network state")
        else:
            bump(RiskLevel.LOW, "read-only network info")
    elif name in MEDIUM_COMMANDS:
        bump(RiskLevel.MEDIUM, f"medium-risk command: {name}")
    elif name in LOW_COMMANDS:
        bump(RiskLevel.LOW, f"low-risk command: {name}")
    else:
        # 未知命令不做乐观假设：默认沙箱执行。
        bump(RiskLevel.MEDIUM, f"unknown command: {name}, default to sandbox")

    # 第八层：写入类副作用兜底（tee、sed -i、in-place 编辑）。
    if _has_tee(tokens):
        bump(RiskLevel.MEDIUM, "tee writes to a file")
    if name == "sed" and _sed_in_place(args):
        bump(RiskLevel.MEDIUM, "sed -i edits files in place")

    # 第九层：敏感路径扫描（非选项参数）。
    for token in _path_like_tokens(args):
        if not is_critical_path(token):
            continue
        if level >= RiskLevel.HIGH:
            # 删除/权限/写入类命令作用于系统敏感路径 → CRITICAL。
            bump(RiskLevel.CRITICAL, f"mutating command targets sensitive system path: {token}")
        elif level == RiskLevel.MEDIUM:
            # 中等风险命令触碰系统目录（如解压到 /etc）→ 升级为 CRITICAL。
            bump(RiskLevel.CRITICAL, f"side-effecting command targets sensitive system path: {token}")
        else:
            # 只读命令读取系统敏感路径（/etc/shadow 等）→ 需要审批。
            bump(RiskLevel.HIGH, f"read access to sensitive system path: {token}")

    return RiskResult(level, reasons)


# ---------------------------------------------------------------------------
# 特殊命令分析
# ---------------------------------------------------------------------------

def _analyze_special_command(name: str, args: list[str], tokens: list[str]) -> RiskResult | None:
    """rm / chmod / dd / Windows 删除命令等：命令 + 参数 + 目标联合判断。"""
    if name == "rm":
        return _analyze_rm(args)
    if name in ("rd", "rmdir"):
        return _analyze_rd(args)
    if name in ("del", "erase"):
        return _analyze_del(args)
    if name in ("chmod", "chown", "chgrp", "icacls", "cacls", "takeown"):
        return _analyze_permission_command(name, args)
    if name == "dd":
        return _analyze_dd(args)
    if name == "remove-item":
        return _analyze_remove_item(tokens)
    return None


def _is_option_token(token: str) -> bool:
    """token 是否为命令行选项：Unix 的 -x/--xx，或 Windows 的 /s、/q、/mir 这类开关。

    Unix 绝对路径（/etc、/home/x、/dev/sda）绝不能被误判为 Windows 开关——
    否则 `rm -rf /etc` 的目标会被当开关跳过，敏感路径检测失效。
    """
    if token.startswith("-"):
        return True
    if not token.startswith("/") or token == "/":
        return False
    # 多段 Unix 路径（/etc/passwd）或系统敏感单段路径（/etc、/bin、/usr…）不是开关。
    if _is_unix_abs_path(token) or is_critical_path(token):
        return False
    # Windows 开关：/s、/q、/mir、/copyall、/grant:r、/exclude:f.txt …
    return bool(re.match(r"^/[a-zA-Z0-9?]{1,7}(?::[^/\\]*)?$", token))


def _flag_words(args: list[str]) -> set[str]:
    return {a.lower() for a in args if _is_option_token(a)}


def _positional_targets(args: list[str]) -> list[str]:
    """去掉选项后剩下的位置参数（目标路径候选）；dd 风格 key=value 也排除。"""
    targets = []
    for token in args:
        if _is_option_token(token):
            continue
        if "=" in token:  # of=/dev/sda、bs=4k 这类键值参数不是路径位置参数
            continue
        targets.append(token)
    return targets


def _analyze_rm(args: list[str]) -> RiskResult:
    flags = _flag_words(args)
    recursive = any("r" in f.lower() for f in flags) or "--recursive" in flags
    force = any("f" in f.lower() for f in flags) or "--force" in flags
    reasons = ["file deletion"]
    if recursive:
        reasons.append("recursive deletion")
    if force:
        reasons.append("force deletion")
    for target in _positional_targets(args):
        if is_critical_path(target):
            return RiskResult(
                RiskLevel.CRITICAL,
                reasons + [f"rm targets sensitive system path: {target}"],
            )
    return RiskResult(RiskLevel.HIGH, reasons)


def _analyze_rd(args: list[str]) -> RiskResult:
    """Windows rd/rmdir：/s 递归删除目录树。"""
    flags = _flag_words(args)
    recursive = "/s" in flags or "-s" in flags
    reasons = ["directory deletion"] + (["recursive deletion"] if recursive else [])
    for target in _positional_targets(args):
        if is_critical_path(target):
            return RiskResult(
                RiskLevel.CRITICAL,
                reasons + [f"rd/rmdir targets sensitive system path: {target}"],
            )
    return RiskResult(RiskLevel.HIGH, reasons)


def _analyze_del(args: list[str]) -> RiskResult:
    """Windows del/erase：删除文件（/s 递归）。"""
    flags = _flag_words(args)
    recursive = "/s" in flags
    reasons = ["file deletion"] + (["recursive deletion"] if recursive else [])
    for target in _positional_targets(args):
        if is_critical_path(target):
            return RiskResult(
                RiskLevel.CRITICAL,
                reasons + [f"del targets sensitive system path: {target}"],
            )
    return RiskResult(RiskLevel.HIGH, reasons)


def _analyze_permission_command(name: str, args: list[str]) -> RiskResult:
    for target in _positional_targets(args):
        if is_critical_path(target):
            return RiskResult(
                RiskLevel.CRITICAL,
                [f"{name} modifies permissions on sensitive system path: {target}"],
            )
    return RiskResult(RiskLevel.HIGH, [f"permission modification: {name}"])


def _analyze_dd(args: list[str]) -> RiskResult:
    for token in args:
        if token.lower().startswith("of="):
            target = token[3:]
            if target.lower() not in _SAFE_REDIRECT_TARGETS and (
                is_critical_path(target) or target.startswith("/dev/") or "physicaldrive" in target.lower()
            ):
                return RiskResult(
                    RiskLevel.CRITICAL,
                    [f"dd writes directly to block device: {target}"],
                )
    return RiskResult(RiskLevel.HIGH, ["raw disk/block write (dd)"])


def _analyze_remove_item(tokens: list[str]) -> RiskResult:
    """PowerShell Remove-Item：-Recurse -Force + 敏感路径 → CRITICAL。"""
    lowered = [t.lower() for t in tokens]
    recursive = any("recurse" in t for t in lowered)
    force = "-force" in lowered
    reasons = ["powershell file deletion"]
    if recursive:
        reasons.append("recursive deletion")
    if force:
        reasons.append("force deletion")
    for token in tokens[1:]:
        if token.startswith("-"):
            continue  # -Path/-LiteralPath 的值作为位置参数同样会被扫到
        candidate = _strip_quotes(token)
        if is_critical_path(candidate):
            return RiskResult(
                RiskLevel.CRITICAL,
                reasons + [f"Remove-Item targets sensitive system path: {candidate}"],
            )
    return RiskResult(RiskLevel.HIGH, reasons)


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def _base_command_name(token: str) -> str:
    """取命令名：basename、去引号、去 .exe 后缀、小写。"""
    token = _strip_quotes(token)
    name = os.path.basename(token.replace("\\", "/"))
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name.lower()


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _first_word(args: list[str]) -> str:
    for token in args:
        if not token.startswith("-"):
            return token.lower()
    return ""


def _has_redirect(tokens: list[str]) -> bool:
    return any(t in _REDIRECT_OPS for t in tokens)


def _has_tee(tokens: list[str]) -> bool:
    return any(_base_command_name(t) == "tee" for t in tokens)


def _sed_in_place(args: list[str]) -> bool:
    for token in args:
        low = token.lower()
        if low in ("-i", "--in-place"):
            return True
        if re.match(r"^-i[a-z.]*$", low) or re.match(r"^--in-place(=|$)", low):
            return True
    return False


def _extract_redirect_targets(tokens: list[str]) -> list[str]:
    """提取 > / >> / 2> 等重定向的目标文件（2>&1 这类 fd 重定向跳过）。"""
    targets: list[str] = []
    for index, token in enumerate(tokens):
        glued = None
        if token in _REDIRECT_OPS:
            if index + 1 < len(tokens):
                glued = tokens[index + 1]
        elif re.match(r"^(?:&|1|2)?>>?", token):
            # 形如 `>file`、`2>file`、`&>file` 的粘连写法
            glued = re.sub(r"^(?:&|1|2)?>>?", "", token)
        if glued and not glued.startswith("&"):
            targets.append(_strip_quotes(glued))
    return targets


def _is_unix_abs_path(token: str) -> bool:
    """/etc、/home/x 这类 Unix 绝对路径；区别于 Windows 的 /s、/q 短开关。"""
    if not token.startswith("/"):
        return False
    if token == "/":
        return True
    # 多段路径（/etc/passwd、/dev/sda）或足够长的单段（/etc、/bin、/usr、/root…）。
    return "/" in token[1:] or len(token) >= 4


def _path_like_tokens(args: list[str]) -> list[str]:
    """位置参数中看起来像路径的 token（绝对路径 / 盘符 / 点或 ~ 开头 / 含分隔符）。"""
    result = []
    for token in _positional_targets(args):
        stripped = _strip_quotes(token)
        if not stripped or stripped.startswith("-"):
            continue
        if (
            stripped == "/"
            or stripped.startswith(("//", "\\\\"))
            or _is_unix_abs_path(stripped)
            or re.match(r"^[a-zA-Z]:[\\/]", stripped)
            or stripped.startswith((".", "~"))
            or "/" in stripped
            or "\\" in stripped
        ):
            result.append(stripped)
    return result


def _extract_substitutions(segment: str) -> list[str]:
    """提取 $(...) 与反引号中的命令文本（括号配对）。"""
    inner_parts: list[str] = []
    i = 0
    while i < len(segment):
        if segment.startswith("$(", i):
            depth = 0
            j = i + 2
            while j < len(segment):
                if segment[j] == "(":
                    depth += 1
                elif segment[j] == ")":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            inner_parts.append(segment[i + 2:j])
            i = j + 1
            continue
        if segment[i] == "`":
            j = segment.find("`", i + 1)
            if j == -1:
                break
            inner_parts.append(segment[i + 1:j])
            i = j + 1
            continue
        i += 1
    return [p for p in inner_parts if p.strip()]


def _extract_nested_script(tokens: list[str]) -> tuple[str, bool] | None:
    """找 `bash -c "..."` / `powershell -Command "..."` / `cmd /c "..."` 的脚本内容。

    返回 (脚本内容, 是否编码不可读)；不是嵌套 shell 调用返回 None。
    """
    for index, token in enumerate(tokens):
        name = _base_command_name(token)
        if name not in SHELL_HOST_COMMANDS:
            continue
        for j in range(index + 1, len(tokens)):
            flag = tokens[j].lower()
            if flag in _SHELL_ENCODED_FLAGS:
                return ("", True)
            if flag in _SHELL_SCRIPT_FLAGS or flag.startswith("-c"):
                if j + 1 < len(tokens):
                    return (_strip_quotes(tokens[j + 1]), False)
                return ("", False)
        # bash script.sh / powershell -File x.ps1：执行外部脚本文件，内容不可见
        if index == 0 or any(_base_command_name(t) in PRIVILEGE_COMMANDS for t in tokens[:index]):
            for t in tokens[index + 1:]:
                stripped = _strip_quotes(t)
                if stripped.endswith((".sh", ".ps1", ".bat", ".cmd", ".py")) and not stripped.startswith("-"):
                    return (f"executes script file: {stripped}", False)
    return None
