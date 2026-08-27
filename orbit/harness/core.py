"""Top-level Orbit harness.

The harness owns runtime governance around the Agent loop: context lifecycle,
permission checks, approval, sandboxed command execution, retries, timeouts,
and structured tracing.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..context import ContextManager, estimate_tokens
from ..tools.base import Tool
from .hooks import HookEvent, HookManager, HookResult
from .permissions import PermissionMode, PermissionSettings, PolicyEngine
from .sandbox import SandboxConfig, SandboxRunner, validate_workspace_path
from .state import HarnessState
from .trace import TraceRecorder


ApprovalCallback = Callable[[str, dict[str, Any], str], bool]


@dataclass
class HarnessConfig:
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    trace_dir: Path | None = None
    test_log_dir: Path | None = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    tool_timeout_seconds: int = 60
    max_retries: int = 0
    sandbox_backend: str = "local"
    docker_image: str = "python:3.13-slim"
    docker_network_enabled: bool = False
    docker_cpus: float = 1.0
    docker_memory: str = "512m"
    docker_pids_limit: int = 256
    docker_read_only_rootfs: bool = True
    docker_seccomp_profile: str = ""


class OrbitHarness:
    """Runtime wrapper that coordinates Orbit execution lifecycle."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        tracer: TraceRecorder | None = None,
    ):
        self.config = config or HarnessConfig()
        self.workspace_root = Path(self.config.workspace_root).expanduser().resolve()
        self.tracer = tracer or TraceRecorder(self.config.trace_dir)
        self.approval_callback = approval_callback
        self._contexts: list[int] = []
        self.state = HarnessState(
            session_id=self.tracer.session_id,
            workspace_root=self.workspace_root,
            trace_path=self.tracer.trace_path,
        )
        self.hooks = HookManager(self.tracer)
        self.policy = PolicyEngine(
            PermissionSettings(mode=self.config.permission_mode),
            workspace_root=self.workspace_root,
        )
        self.sandbox = SandboxRunner(
            SandboxConfig(
                backend=self.config.sandbox_backend,
                docker_image=self.config.docker_image,
                network_enabled=self.config.docker_network_enabled,
                cpu_limit=self.config.docker_cpus,
                memory_limit=self.config.docker_memory,
                pids_limit=self.config.docker_pids_limit,
                read_only_rootfs=self.config.docker_read_only_rootfs,
                seccomp_profile=self.config.docker_seccomp_profile,
                test_log_dir=self.config.test_log_dir,
            ),
            workspace_root=self.workspace_root,
        )
        self._closed = False
        self.tracer.record("startup", "harness.core", "harness_created", {
            "workspace_root": str(self.workspace_root),
            "permission_mode": self.config.permission_mode.value,
            "tool_timeout_seconds": self.config.tool_timeout_seconds,
            "max_retries": self.config.max_retries,
            "sandbox_backend": self.config.sandbox_backend,
            "docker_image": self.config.docker_image,
            "docker_network_enabled": self.config.docker_network_enabled,
            "docker_cpus": self.config.docker_cpus,
            "docker_memory": self.config.docker_memory,
            "docker_pids_limit": self.config.docker_pids_limit,
            "docker_read_only_rootfs": self.config.docker_read_only_rootfs,
            "docker_seccomp_profile": self.config.docker_seccomp_profile,
            "trace_dir": str(self.tracer.trace_dir),
            "test_log_dir": str(self.config.test_log_dir) if self.config.test_log_dir else None,
        })
        self._install_default_hooks()

    @classmethod
    def default(cls) -> "OrbitHarness":
        return cls(HarnessConfig(permission_mode=PermissionMode.FULL_AUTO))

    def create_agent(self, llm, tools=None, max_context_tokens: int = 128_000, max_rounds: int = 50):
        from ..agent import Agent

        self.tracer.record("agent", "harness.core", "create_agent", {
            "max_context_tokens": max_context_tokens,
            "max_rounds": max_rounds,
            "tool_count": len(tools) if tools is not None else None,
        })
        return Agent(
            llm=llm,
            tools=tools,
            max_context_tokens=max_context_tokens,
            max_rounds=max_rounds,
            harness=self,
        )

    def create_context(self, max_tokens: int) -> ContextManager:
        context = ContextManager(max_tokens=max_tokens)
        self._contexts.append(id(context))
        self.state.context_ids.append(id(context))
        self.tracer.record("context", "harness.context", "context_created", {
            "context_id": id(context),
            "max_tokens": max_tokens,
        })
        return context

    def run_chat(self, agent, user_input: str, on_token=None, on_tool=None, on_tool_result=None) -> str:
        self.tracer.record("execution", "harness.core", "execution_started", {
            "input_chars": len(user_input),
            "state": self.state.snapshot(),
        })
        try:
            return agent.chat(user_input, on_token=on_token, on_tool=on_tool, on_tool_result=on_tool_result)
        except Exception as exc:
            self.tracer.record_error("execution", "harness.core", "execution_failed", exc)
            raise
        finally:
            self.tracer.record("execution", "harness.core", "execution_finished")

    def start_chat(self, user_input: str) -> None:
        self.state.current_input = user_input
        self.hooks.trigger(HookEvent.USER_PROMPT_SUBMIT, {
            "input": user_input,
            "workspace_root": str(self.workspace_root),
            "state": self.state.snapshot(),
        })
        self.tracer.record("chat", "agent", "chat_started", {
            "input_chars": len(user_input),
        })

    def finish_chat(self, result: str, *, reason: str) -> None:
        self.hooks.trigger(HookEvent.STOP, {
            "reason": reason,
            "result": result,
            "state": self.state.snapshot(),
        })
        self.tracer.record("chat", "agent", "chat_finished", {
            "reason": reason,
            "result_chars": len(result),
            "state": self.state.snapshot(),
        })

    def record_round_start(self, round_index: int, max_rounds: int, message_count: int) -> None:
        self.state.round_count = round_index
        self.tracer.record("loop", "agent", "round_started", {
            "round": round_index,
            "max_rounds": max_rounds,
            "message_count": message_count,
        })

    def compress_context_if_needed(self, context: ContextManager, messages: list[dict], llm) -> bool:
        before_tokens = estimate_tokens(messages)
        self.tracer.record("context", "harness.context", "context_compress_check_started", {
            "message_count": len(messages),
            "estimated_tokens": before_tokens,
        })
        started = time.monotonic()
        compressed = False
        try:
            compressed = llm is not None and context.maybe_compress(messages, llm)
            return compressed
        except Exception as exc:
            self.tracer.record_error("context", "harness.context", "context_compress_failed", exc)
            raise
        finally:
            self.tracer.record("context", "harness.context", "context_compress_check_finished", {
                "context_id": id(context),
                "message_count": len(messages),
                "estimated_tokens": estimate_tokens(messages),
                "compressed": compressed,
            }, duration_ms=_elapsed_ms(started))
            self.tracer.record("context", "harness.context", "context_updated", {
                "context_id": id(context),
                "message_count": len(messages),
            })
            self.hooks.trigger(HookEvent.CONTEXT_UPDATE, {
                "context_id": id(context),
                "message_count": len(messages),
                "compressed": compressed,
                "state": self.state.snapshot(),
            })

    def attach_context(self, context: ContextManager) -> None:
        self.tracer.record("context", "harness.context", "context_attached", {
            "context_id": id(context),
            "max_tokens": context.max_tokens,
        })

    def run_llm_call(self, call: Callable[[], Any], *, round_index: int) -> Any:
        pre_result = self.hooks.trigger(HookEvent.PRE_LLM_CALL, {
            "round": round_index,
            "state": self.state.snapshot(),
        })
        if pre_result.blocked:
            raise RuntimeError(pre_result.reason or "PreLLMCall hook blocked LLM call")
        self.tracer.record("llm", "agent", "llm_call_started", {"round": round_index})
        started = time.monotonic()
        try:
            result = call()
            self.hooks.trigger(HookEvent.POST_LLM_CALL, {
                "round": round_index,
                "tool_call_count": len(getattr(result, "tool_calls", []) or []),
                "content_chars": len(getattr(result, "content", "") or ""),
                "state": self.state.snapshot(),
            })
            self.tracer.record("llm", "agent", "llm_call_finished", {
                "round": round_index,
                "tool_call_count": len(getattr(result, "tool_calls", []) or []),
                "content_chars": len(getattr(result, "content", "") or ""),
            }, duration_ms=_elapsed_ms(started))
            return result
        except Exception as exc:
            self.tracer.record_error("llm", "agent", "llm_call_failed", exc, {"round": round_index})
            raise

    def execute_tool_call(self, tool: Tool, tool_call) -> str:
        tool_name = getattr(tool_call, "name", "")
        arguments = dict(getattr(tool_call, "arguments", {}) or {})
        self.state.tool_call_count += 1
        self.tracer.record("tool", "harness.core", "tool_call_received", {
            "tool_name": tool_name,
            "tool_call_id": getattr(tool_call, "id", ""),
            "arguments": _redact_arguments(arguments),
        })

        bind_error = self._validate_arguments(tool, arguments)
        if bind_error:
            return bind_error

        normalized_path = self._resolve_tool_path(arguments)
        command = arguments.get("command") if isinstance(arguments.get("command"), str) else None
        payload = {
            "tool": tool,
            "tool_name": tool_name,
            "tool_call_id": getattr(tool_call, "id", ""),
            "arguments": arguments,
            "redacted_arguments": _redact_arguments(arguments),
            "path": normalized_path,
            "command": command,
            "state": self.state.snapshot(),
        }
        blocked = self.hooks.trigger(HookEvent.PRE_TOOL_USE, payload)
        if blocked.blocked:
            if blocked.reason.startswith("approval denied: "):
                return f"Blocked by harness approval: {blocked.reason.removeprefix('approval denied: ')}"
            return f"Blocked by harness: {blocked.reason}"

        output = self._execute_with_retry(tool, arguments)
        self.hooks.trigger(HookEvent.POST_TOOL_USE, {
            **payload,
            "output": output,
            "output_chars": len(output),
            "state": self.state.snapshot(),
        })
        return output

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> str:
        try:
            inspect.signature(tool.execute).bind(**arguments)
        except TypeError as exc:
            self.tracer.record("tool", "harness.core", "tool_arguments_invalid", {
                "tool_name": tool.name,
                "error": str(exc),
            }, status="error")
            return f"Error: bad arguments for {tool.name}: {exc}"
        return ""

    def _resolve_tool_path(self, arguments: dict[str, Any]) -> Path | None:
        for key in ("file_path", "path", "root"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = self.workspace_root / path
                return path
        return None

    def _approve(self, tool_name: str, arguments: dict[str, Any], reason: str) -> bool:
        self.tracer.record("approval", "harness.approval", "approval_requested", {
            "tool_name": tool_name,
            "reason": reason,
            "arguments": _redact_arguments(arguments),
        })
        approved = bool(self.approval_callback and self.approval_callback(tool_name, arguments, reason))
        self.tracer.record("approval", "harness.approval", "approval_resolved", {
            "tool_name": tool_name,
            "approved": approved,
        }, status="ok" if approved else "blocked")
        return approved

    def _execute_with_retry(self, tool: Tool, arguments: dict[str, Any]) -> str:
        attempts = max(1, self.config.max_retries + 1)
        last_result = ""
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            details = {
                "tool_name": tool.name,
                "attempt": attempt,
                "max_attempts": attempts,
                "sandbox_backend": self.config.sandbox_backend,
            }
            if tool.name == "bash" and isinstance(arguments.get("command"), str):
                details["command"] = arguments["command"]
            self.tracer.record("tool", "harness.core", "tool_execution_started", {
                **details,
            })
            try:
                result = self._invoke_tool(tool, arguments)
                self.tracer.record("tool", "harness.core", "tool_execution_finished", {
                    "tool_name": tool.name,
                    "attempt": attempt,
                    "output_chars": len(result),
                }, duration_ms=_elapsed_ms(started))
                return result
            except TimeoutError as exc:
                last_result = f"Error executing {tool.name}: timed out after {self.config.tool_timeout_seconds}s"
                self.tracer.record_error("tool", "harness.core", "tool_execution_timeout", exc, {
                    "tool_name": tool.name,
                    "attempt": attempt,
                })
            except Exception as exc:  # noqa: BLE001
                last_result = f"Error executing {tool.name}: {exc}"
                self.tracer.record_error("tool", "harness.core", "tool_execution_failed", exc, {
                    "tool_name": tool.name,
                    "attempt": attempt,
                })
        return last_result

    def _invoke_tool(self, tool: Tool, arguments: dict[str, Any]) -> str:
        if tool.name == "bash":
            command = str(arguments["command"])
            requested_timeout = int(arguments.get("timeout", self.config.tool_timeout_seconds))
            timeout = min(requested_timeout, self.config.tool_timeout_seconds)
            output = self.sandbox.run_bash(command, timeout)
            if self.sandbox.last_test_log_path is not None:
                self.tracer.record("tool", "harness.sandbox", "test_log_saved", {
                    "tool_name": tool.name,
                    "command": command,
                    "test_log_path": str(self.sandbox.last_test_log_path),
                })
                self.sandbox.last_test_log_path = None
            return output
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.execute, **arguments)
        try:
            return future.result(timeout=self.config.tool_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError from exc
        finally:
            if future.done():
                pool.shutdown(wait=False, cancel_futures=True)

    def save_trace(self) -> Path:
        path = self.tracer.save()
        self.state.trace_path = path
        self.tracer.record("trace", "harness.trace", "trace_saved", {
            "trace_path": str(path),
            "event_count": len(self.tracer.events),
            "state": self.state.snapshot(),
        })
        self.tracer.save()
        return path

    def close(self) -> Path:
        if not self._closed:
            self._closed = True
            self.hooks.trigger(HookEvent.SHUTDOWN, {
                "state": self.state.snapshot(),
            })
            for context_id in self._contexts:
                self.tracer.record("context", "harness.context", "context_destroyed", {
                    "context_id": context_id,
                })
            self.state.closed = True
            self.tracer.record("shutdown", "harness.core", "harness_closed", {
                "workspace_root": str(self.workspace_root),
                "state": self.state.snapshot(),
            })
        return self.save_trace()

    def _install_default_hooks(self) -> None:
        self.hooks.register(HookEvent.PRE_TOOL_USE, self._permission_hook)
        self.hooks.register(HookEvent.PRE_TOOL_USE, self._tool_log_hook)
        self.hooks.register(HookEvent.POST_TOOL_USE, self._large_output_hook)
        self.hooks.register(HookEvent.STOP, self._stop_summary_hook)

    def _permission_hook(self, payload: dict[str, Any]) -> HookResult:
        tool_name = str(payload.get("tool_name") or "")
        arguments = dict(payload.get("arguments") or {})
        normalized_path = payload.get("path")
        if normalized_path is not None:
            allowed, reason, resolved = validate_workspace_path(normalized_path, self.workspace_root)
            self.tracer.record("permission", "harness.sandbox", "workspace_path_checked", {
                "tool_name": tool_name,
                "path": str(resolved),
                "allowed": allowed,
                "reason": reason,
            }, status="ok" if allowed else "blocked")
            if not allowed:
                return HookResult(blocked=True, reason=reason)
            normalized_path = resolved

        command = payload.get("command")
        command = command if isinstance(command, str) else None
        self.state.permission_check_count += 1
        decision = self.policy.evaluate(
            tool_name,
            arguments,
            tool_read_only=bool(getattr(payload.get("tool"), "read_only", False)),
            file_path=normalized_path if isinstance(normalized_path, Path) else None,
            command=command,
        )
        self.tracer.record("permission", "harness.permissions", "permission_checked", {
            "tool_name": tool_name,
            "allowed": decision.allowed,
            "requires_approval": decision.requires_approval,
            "reason": decision.reason,
        }, status="ok" if decision.allowed else ("approval_required" if decision.requires_approval else "blocked"))

        if decision.allowed:
            return HookResult()
        if not decision.requires_approval:
            return HookResult(blocked=True, reason=decision.reason)
        if self._approve(tool_name, arguments, decision.reason):
            return HookResult()
        return HookResult(blocked=True, reason=f"approval denied: {decision.reason}")

    def _tool_log_hook(self, payload: dict[str, Any]) -> None:
        self.tracer.record("tool", "harness.hooks", "tool_log_hook", {
            "tool_name": payload.get("tool_name"),
            "arguments": payload.get("redacted_arguments", {}),
        })
        return None

    def _large_output_hook(self, payload: dict[str, Any]) -> None:
        output_chars = int(payload.get("output_chars") or 0)
        if output_chars > 100_000:
            self.tracer.record("tool", "harness.hooks", "large_output_detected", {
                "tool_name": payload.get("tool_name"),
                "output_chars": output_chars,
            }, status="warning")
        return None

    def _stop_summary_hook(self, payload: dict[str, Any]) -> None:
        self.tracer.record("shutdown", "harness.hooks", "stop_summary", {
            "reason": payload.get("reason"),
            "tool_call_count": self.state.tool_call_count,
            "round_count": self.state.round_count,
            "permission_check_count": self.state.permission_check_count,
        })
        return None


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if any(secret in key.lower() for secret in ("key", "token", "secret", "password")):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted
