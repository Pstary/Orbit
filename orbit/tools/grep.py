"""Content search with regex support."""

import re
from pathlib import Path
from typing import ClassVar

from .base import Tool
# 在项目文件里按正则查找内容，并返回文件路径、行号和命中行。
# skip these dirs to avoid noise
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with regex. "
        "Returns matching lines with file path and line number."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: cwd)",
            },
            "include": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py')",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str:
        try:
            # 编译正则表达式模式，用于后续匹配。
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        # 将用户传入的搜索路径规范化成绝对路径，并展开 `~` 这类用户目录写法。
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"Error: {path} not found"

        # 如果搜索目标本身是文件，就只扫描这个文件；否则递归收集目录下符合 include 的文件。
        if base.is_file():
            files = [base]
            scan_truncated = False
        else:
            files, scan_truncated = self._walk(base, include)

        # 目录扫描最多收集5000个文件，结果匹配最多返回200行，避免一次搜索输出过大。
        scan_limit_msg = "... (5000 file scan limit reached; results may be incomplete)"
        matches = []
        for fp in files:
            try:
                # 以UTF-8读取文本；无法解码的字节会被忽略，读取失败的文件直接跳过。
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    # 每条命中结果带上文件路径、行号和原始行内容，方便调用方定位。
                    matches.append(f"{fp}:{lineno}: {line.rstrip()}")
                    if len(matches) >= 200:
                        matches.append("... (200 match limit reached)")
                        if scan_truncated:
                            matches.append(scan_limit_msg)
                        return "\n".join(matches)

        # 有命中时返回全部命中；如果文件扫描被截断，需要显式提示结果可能不完整。
        if matches:
            if scan_truncated:
                matches.append(scan_limit_msg)
            return "\n".join(matches)
        if scan_truncated:
            return f"No matches found in scanned files.\n{scan_limit_msg}"
        return "No matches found."

    @staticmethod
    def _walk(root: Path, include: str | None) -> tuple[list[Path], bool]:
        """递归遍历目录树，并跳过常见噪声目录。"""
        results = []
        truncated = False
        for item in root.rglob(include or "*"):
            # 只判断搜索根目录内部的路径片段，避免根目录祖先中包含build等名字时误跳过整棵树。
            if any(part in _SKIP_DIRS for part in item.relative_to(root).parts):
                continue
            if item.is_file():
                results.append(item)
            if len(results) >= 5000:
                # 达到扫描上限后停止继续遍历，并把截断状态返回给调用方。
                truncated = True
                break
        return results, truncated
