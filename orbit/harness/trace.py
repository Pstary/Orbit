"""Structured execution tracing for Orbit."""

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


# 单条内容字段（用户query、LLM输出、工具结果等）写入trace时的字符上限。
# trace 文件用于事后分析，保留头尾比只留长度有用，但全文落盘会让文件无限膨胀，
# 因此超长内容按"头70% + 尾30%"截断并标注省略的字符数（错误信息常在尾部）。
TRACE_CONTENT_LIMIT = 8_000


def truncate_for_trace(text: object, limit: int = TRACE_CONTENT_LIMIT) -> str:
    """把长文本截成 head+tail 预览，保证 trace 可读且不无限膨胀。"""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return (
        text[:head]
        + f"\n... [truncated {len(text) - limit} chars in trace] ...\n"
        + text[-tail:]
    )


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
            "summary": self._build_summary(),
            "events": [asdict(event) for event in self.events],
        }
        self.trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.trace_path

    def _build_summary(self) -> dict:
        """扫描全部事件，汇总 token/费用/调用次数/错误数/总耗时，便于一眼看全局。"""
        prompt_tokens = 0
        completion_tokens = 0
        llm_calls = 0
        tool_calls = 0
        error_count = 0
        warning_count = 0
        cost_usd = 0.0
        models: set[str] = set()
        for event in self.events:
            details = event.details or {}
            if event.action == "llm_call_finished":
                llm_calls += 1
                prompt_tokens += int(details.get("prompt_tokens") or 0)
                completion_tokens += int(details.get("completion_tokens") or 0)
                call_cost = details.get("call_cost_usd")
                if isinstance(call_cost, (int, float)):
                    cost_usd += float(call_cost)
                model = details.get("model")
                if model:
                    models.add(str(model))
            elif event.action == "tool_call_received":
                tool_calls += 1
            if event.status == "error":
                error_count += 1
            elif event.status == "warning":
                warning_count += 1
        duration_ms = None
        try:
            started = datetime.fromisoformat(self.started_at)
            duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000, 3)
        except (TypeError, ValueError):
            pass
        return {
            "duration_ms": duration_ms,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": round(cost_usd, 6) if cost_usd else None,
            "models": sorted(models),
            "error_count": error_count,
            "warning_count": warning_count,
        }
