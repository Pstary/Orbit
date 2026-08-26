"""Structured execution tracing for CoreCoder."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def default_trace_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "trace" / _date_dir()


@dataclass
class TraceEvent:
    timestamp: str
    stage: str
    module: str
    action: str
    status: str = "ok"
    details: dict = field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None
    stack: str | None = None


class TraceRecorder:
    """Collects runtime events and persists them as analysis-friendly JSON."""

    def __init__(self, trace_dir: str | Path | None = None, session_id: str | None = None):
        self.session_id = session_id or uuid4().hex
        self.started_at = _utc_now()
        self.trace_dir = Path(trace_dir).expanduser().resolve() if trace_dir else default_trace_dir()
        safe_started = self.started_at.replace(":", "").replace("+", "Z")
        self.trace_path = self.trace_dir / f"trace-{safe_started}-{self.session_id[:8]}.json"
        self.events: list[TraceEvent] = []
        self.record("startup", "harness.trace", "trace_recorder_created", {
            "session_id": self.session_id,
            "trace_path": str(self.trace_path),
        })

    def record(
        self,
        stage: str,
        module: str,
        action: str,
        details: dict | None = None,
        *,
        status: str = "ok",
        duration_ms: float | None = None,
    ) -> None:
        self.events.append(TraceEvent(
            timestamp=_utc_now(),
            stage=stage,
            module=module,
            action=action,
            status=status,
            details=details or {},
            duration_ms=duration_ms,
        ))

    def record_error(
        self,
        stage: str,
        module: str,
        action: str,
        error: BaseException,
        details: dict | None = None,
    ) -> None:
        self.events.append(TraceEvent(
            timestamp=_utc_now(),
            stage=stage,
            module=module,
            action=action,
            status="error",
            details=details or {},
            error=f"{type(error).__name__}: {error}",
            stack=traceback.format_exc(),
        ))

    def save(self) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "saved_at": _utc_now(),
            "event_count": len(self.events),
            "events": [asdict(event) for event in self.events],
        }
        self.trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.trace_path
