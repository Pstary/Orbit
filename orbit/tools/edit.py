"""Search-and-replace file editing (Claude Code's key innovation).

The core idea: instead of sending whole-file rewrites or line-number patches,
the LLM specifies an *exact* substring to find and its replacement. The
substring must appear exactly once in the file, which eliminates ambiguity
and makes edits safe and reviewable.
"""

import difflib
from pathlib import Path
from typing import ClassVar

from .base import Tool

# track files changed this session for /diff
_changed_files: set[str] = set()


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match. "
        "old_string must appear exactly once in the file for safety. "
        "Include enough surrounding context to ensure uniqueness."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, file_path: str, old_string: str, new_string: str) -> str:
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"

            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: {file_path} is not a UTF-8 text file (edit_file only edits text files)"
            # 检查old_string是否在文件中出现一次
            occurrences = content.count(old_string)

            if occurrences == 0:
                # 如果old_string不在文件中，返回错误信息
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return (
                    f"Error: old_string not found in {file_path}.\n"
                    f"File starts with:\n{preview}"
                )
            if occurrences > 1:
                # 如果old_string在文件中出现多次，返回错误信息  
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    f"Include more surrounding lines to make it unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding="utf-8")
            _changed_files.add(str(p))

            # generate a unified diff so the user/LLM can see exactly what changed
            # 生成一个紧凑的统一差异，显示文件内容变化的详细信息。
            # 包括文件名、行号、旧值和新值。
            diff = _unified_diff(content, new_content, str(p))
            return f"Edited {file_path}\n{diff}"
        except Exception as e:  # noqa: BLE001
            # boundary: the agent gets an error string, not a traceback
            return f"Error: {e}"


def _unified_diff(old: str, new: str, filename: str, context: int = 3) -> str:
    """Generate a compact unified diff between old and new file content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    # 使用标准 unified diff 格式生成补丁文本，方便调用方直接看到变更前后的行。
    # fromfile/tofile 分别标记旧文件和新文件，n 控制每个变更块保留多少行上下文。
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context,
    )
    result = "".join(diff)
    # truncate enormous diffs
    if len(result) > 3000:
        result = result[:2500] + "\n... (diff truncated)\n"
    return result   
