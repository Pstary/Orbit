"""File pattern matching."""

from pathlib import Path
from typing import ClassVar

from .base import Tool


class GlobTool(Tool):
    name = "glob"
    read_only = True
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            # 将搜索目录规范化为绝对路径，并展开`~`这类用户目录写法。
            base = Path(path).expanduser().resolve()
            if not base.exists():
                return f"Error: {path} not found"
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            # 在指定目录下按glob模式匹配文件；`**`可以递归匹配多级目录。
            hits = list(base.glob(pattern))
            # 按文件修改时间倒序排序，让最近变更的文件排在前面。
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            # 最多返回前100条，避免一次glob结果过大影响上层Agent阅读。
            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:  # noqa: BLE001
            # 工具边界统一返回错误文本，不把Python异常栈直接抛给上层Agent。
            return f"Error: {e}"
