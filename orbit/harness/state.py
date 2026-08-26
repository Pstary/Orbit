"""Runtime state owned by the Orbit harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HarnessState:
    session_id: str
    workspace_root: Path
    started_at: str = field(default_factory=_utc_now)
    current_input: str = ""
    round_count: int = 0
    tool_call_count: int = 0
    permission_check_count: int = 0
    context_ids: list[int] = field(default_factory=list)
    trace_path: Path | None = None
    closed: bool = False

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "started_at": self.started_at,
            "current_input_chars": len(self.current_input),
            "round_count": self.round_count,
            "tool_call_count": self.tool_call_count,
            "permission_check_count": self.permission_check_count,
            "context_count": len(self.context_ids),
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "closed": self.closed,
        }
