"""File creation / overwrite."""

from pathlib import Path
from typing import ClassVar

from .base import Tool
from .edit import _changed_files
from .runtime import check_deadline, file_mutation_lock


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str, content: str) -> str:
        with file_mutation_lock:
            try:
                p = Path(file_path).expanduser().resolve()
                check_deadline()
                p.parent.mkdir(parents=True, exist_ok=True)
                check_deadline()
                p.write_text(content, encoding="utf-8")
                _changed_files.add(str(p))
                n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
                return f"Wrote {n_lines} lines to {file_path}"
            except Exception as e:  # noqa: BLE001
                # boundary: the agent gets an error string, not a traceback
                return f"Error: {e}"
