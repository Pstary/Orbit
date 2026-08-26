"""Hook kernel for CoreCoder harness lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .trace import TraceRecorder


class HookEvent(str, Enum):
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_LLM_CALL = "PreLLMCall"
    POST_LLM_CALL = "PostLLMCall"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    CONTEXT_UPDATE = "ContextUpdate"
    STOP = "Stop"
    SHUTDOWN = "Shutdown"


@dataclass(frozen=True)
class HookResult:
    blocked: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[dict[str, Any]], HookResult | str | None]


class HookManager:
    """Runs lifecycle callbacks at fixed harness extension points."""

    def __init__(self, tracer: TraceRecorder):
        self._tracer = tracer
        self._hooks: dict[HookEvent, list[HookCallback]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        self._hooks[event].append(callback)
        self._tracer.record("hook", "harness.hooks", "hook_registered", {
            "event": event.value,
            "callback": getattr(callback, "__name__", repr(callback)),
        })

    def trigger(self, event: HookEvent, payload: dict[str, Any] | None = None) -> HookResult:
        active_payload = payload or {}
        callbacks = self._hooks.get(event, [])
        self._tracer.record("hook", "harness.hooks", "hook_trigger_started", {
            "event": event.value,
            "callback_count": len(callbacks),
        })
        for callback in callbacks:
            name = getattr(callback, "__name__", repr(callback))
            try:
                raw = callback(active_payload)
            except Exception as exc:  # noqa: BLE001
                self._tracer.record_error("hook", "harness.hooks", "hook_failed", exc, {
                    "event": event.value,
                    "callback": name,
                })
                return HookResult(blocked=True, reason=f"Hook {name} failed: {exc}")
            result = _normalize_hook_result(raw)
            self._tracer.record("hook", "harness.hooks", "hook_callback_finished", {
                "event": event.value,
                "callback": name,
                "blocked": result.blocked,
                "reason": result.reason,
            }, status="blocked" if result.blocked else "ok")
            if result.blocked:
                return result
        self._tracer.record("hook", "harness.hooks", "hook_trigger_finished", {
            "event": event.value,
        })
        return HookResult()


def _normalize_hook_result(raw: HookResult | str | None) -> HookResult:
    if raw is None:
        return HookResult()
    if isinstance(raw, HookResult):
        return raw
    if isinstance(raw, str) and raw:
        return HookResult(blocked=True, reason=raw)
    return HookResult()
