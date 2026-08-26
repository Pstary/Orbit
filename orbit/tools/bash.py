"""Shell command execution with safety checks.

Claude Code's BashTool is 1,143 lines. This is the distilled version:
- Output capture with truncation (head+tail preserved)
- Timeout support
- Dangerous command detection
- Working directory tracking (cd awareness)
"""

# bash 工具实现 ，给 Agent 用来执行终端命令。

import os
import re
import subprocess
import threading
from typing import ClassVar

from .base import Tool

# Track cwd across commands (Claude Code does this too). Thread-local, so that
# when the agent executes tools in parallel two bash calls never race on one
# shared global: each worker thread carries its own cwd. See article 05.
_local = threading.local()

# patterns that could wreck the filesystem or leak secrets
_DANGEROUS_PATTERNS = [
    # recursive delete aimed at root/home (force flag optional)
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    # recursive (-r/-R) and force (-f) flags together, in any order or spacing
    (r"\brm\b(?=(?:.*\s)?-\w*[rR])(?=(?:.*\s)?-\w*f)", "force recursive delete"),
    # the same, written with long-form flags
    (r"\brm\b.*--recursive\b.*--force\b|\brm\b.*--force\b.*--recursive\b", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe curl to shell"),
    (r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe wget to shell"),
]


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 120) -> str:
        # safety check
        warning = _check_dangerous(command)
        if warning:
            return f"⚠ Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # use this thread's own tracked working directory
        cwd = getattr(_local, "cwd", None) or os.getcwd()

        try:
            proc = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                # 表示“让这个shell命令在哪个目录下执行”。
                cwd=cwd,
            )

            # 只有命令成功执行后才更新记录的 cwd，避免失败的 `cd` 污染下一次命令的目录。
            if proc.returncode == 0:
                _update_cwd(command, cwd)

            # 先返回 stdout，再把 stderr 和退出码拼到同一段文本里，保证调用方始终拿到
            # 一个普通文本结果。
            out = proc.stdout
            if proc.stderr:
                out += f"\n[stderr]\n{proc.stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"

            # 对超长输出做截断，但保留开头和结尾：命令上下文通常在开头，错误细节通常在结尾。
            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            # 超时也作为普通文本返回，避免工具调用在传输层失败，便于上层继续分析。
            return f"Error: timed out after {timeout}s"
        except Exception as e:  # noqa: BLE001
            # 其他 OS 层异常（例如进程启动失败）同样转成文本返回。
            return f"Error running command: {e}"


def _check_dangerous(cmd: str) -> str | None:
    """Return a warning string if the command looks destructive, else None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    """按线程追踪cd命令导致的目录变化。"""
    # 逐段处理`&&`串联的命令，只追踪其中的`cd`片段。
    # 相对路径基于上一次`cd`后的目录继续解析，所以`cd a && cd b`最终会落到 a/b。
    running = current_cwd
    changed = False
    for part in command.split("&&"):
        part = part.strip()
        if part.startswith("cd "):
            # 提取`cd`目标目录，并兼容简单的单/双引号包裹。
            target = part[3:].strip().strip("'\"")
            if target:
                new_dir = os.path.normpath(os.path.join(running, os.path.expanduser(target)))
                if os.path.isdir(new_dir):
                    # 只在目标目录真实存在时更新，避免把无效路径写入线程本地 cwd。
                    running = new_dir
                    changed = True
    if changed:
        # 将最终目录写回线程本地状态，供下一次 bash 工具调用复用。
        _local.cwd = running
