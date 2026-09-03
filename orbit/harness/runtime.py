"""Harness-owned conversations, execution scheduling and memory lifecycle."""

from __future__ import annotations

import concurrent.futures
import copy
import threading
import time
from dataclasses import dataclass, field

from ..cancellation import CancellationToken, cancellation_token
from ..context import ContextManager
from ..memory import MemoryManager
from ..prompt import system_prompt
from ..session import list_sessions, load_session, save_session
from ..skill_registry import default_skills_dir
from ..tools import get_default_tools
from ..tools.agent import AgentTool
from ..tools.base import Tool


@dataclass
class AgentRuntime:
    tools: list[Tool]
    context: ContextManager
    max_rounds: int
    system: str
    memory: MemoryManager | None
    messages: list[dict] = field(default_factory=list)
    memory_block: str = ""
    chat_lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_token: CancellationToken | None = None

    @property
    def tool_by_name(self) -> dict[str, Tool]:
        return {tool.name: tool for tool in self.tools}


class RuntimeManager:
    """Conversation orchestration mixed into OrbitHarness.

    Agents retain only their model connection. All mutable runtime state lives
    here, separately for each agent, including agents sharing one harness.
    """

    def _init_runtime(self) -> None:
        self._runtimes: dict[object, AgentRuntime] = {}
        self._memory_notify = None

    def register_agent(self, agent, *, tools, max_context_tokens, max_rounds, memory_enabled) -> None:
        if self._closed:
            raise RuntimeError("Harness is closed")
        if tools is None:
            tools = get_default_tools(
                include_mcp=self.config.mcp_enabled,
                mcp_config_path=self.config.mcp_config_file,
                include_skills=self.config.skills_enabled,
                skills_dir=self.config.skills_dir or default_skills_dir(self.workspace_root),
            )
        # Delegation tools carry a binding, so never share that binding between
        # agents, even when a caller supplies the same tool list twice.
        tools = [copy.copy(tool) if isinstance(tool, AgentTool) else tool for tool in tools]
        context = self.create_context(max_context_tokens)
        self.attach_context(context)
        enabled = self.config.memory_enabled if memory_enabled is None else memory_enabled
        memory = MemoryManager(
            self.workspace_root, tracer=self.tracer, memory_dir=self.memory_dir,
            notify=self._memory_notify,
        ) if enabled else None
        self._runtimes[agent] = AgentRuntime(
            tools=tools, context=context, max_rounds=max_rounds,
            system=system_prompt(tools, workspace_root=self.workspace_root), memory=memory,
        )
        for tool in tools:
            if isinstance(tool, AgentTool):
                tool._parent_agent = agent
        self.tracer.record("agent", "harness.runtime", "agent_initialized", {
            "tool_names": [tool.name for tool in tools], "max_rounds": max_rounds,
            "max_context_tokens": max_context_tokens, "model": getattr(agent.llm, "model", None),
            "memory_enabled": enabled,
        })

    def runtime_for(self, agent) -> AgentRuntime:
        try:
            return self._runtimes[agent]
        except KeyError as exc:
            raise ValueError("Agent does not belong to this harness") from exc

    def full_messages(self, agent) -> list[dict]:
        runtime = self.runtime_for(agent)
        content = runtime.system
        if runtime.memory_block:
            content += "\n\n" + runtime.memory_block
        return [{"role": "system", "content": content}] + runtime.messages

    def tool_schemas(self, agent) -> list[dict]:
        return [tool.schema() for tool in self.runtime_for(agent).tools]

    def _run_agent(self, agent, user_input, on_token=None, on_tool=None, on_tool_result=None):
        runtime = self.runtime_for(agent)
        if not runtime.chat_lock.acquire(blocking=False):
            raise RuntimeError("A chat is already running for this agent")
        runtime.cancel_token = CancellationToken()
        cancel_context = cancellation_token.set(runtime.cancel_token)
        try:
            self.start_chat(user_input)
            runtime.messages.append({"role": "user", "content": user_input})
            self.compact(agent)
            runtime.memory_block = self.recall_memories(agent)
            for round_index in range(1, runtime.max_rounds + 1):
                runtime.cancel_token.raise_if_cancelled()
                self.record_round_start(round_index, runtime.max_rounds, len(runtime.messages))
                response = self.run_llm_call(
                    lambda: agent.respond(self.full_messages(agent), self.tool_schemas(agent), on_token),
                    round_index=round_index, llm=agent.llm,
                )
                runtime.messages.append(response.message)
                if not response.tool_calls:
                    self.finish_chat(response.content, reason="assistant_final")
                    self.extract_memories(agent)
                    return response.content
                try:
                    self.exec_tools(
                        agent, response.tool_calls, on_tool, on_tool_result,
                        _record_messages=True,
                    )
                except BaseException:
                    self.answer_pending_tool_calls(agent, response.tool_calls)
                    raise
                self.compact(agent)
            result = "(reached maximum tool-call rounds)"
            self.finish_chat(result, reason="max_rounds")
            self.extract_memories(agent)
            return result
        finally:
            cancellation_token.reset(cancel_context)
            runtime.cancel_token = None
            runtime.chat_lock.release()

    def interrupt(self, agent) -> bool:
        """Request cancellation of the active turn without discarding its history."""
        runtime = self.runtime_for(agent)
        token = runtime.cancel_token
        if token is None:
            return False
        token.cancel()
        self.tracer.record("execution", "harness.runtime", "execution_interrupt_requested", {
            "state": self.state.snapshot(),
        }, status="warning")
        return True

    def compact(self, agent) -> bool:
        runtime = self.runtime_for(agent)
        return self.compress_context_if_needed(runtime.context, runtime.messages, agent.llm)

    def recall_memories(self, agent) -> str:
        runtime = self.runtime_for(agent)
        if runtime.memory is None:
            return ""
        try:
            return runtime.memory.recall_block(runtime.messages, agent.llm)
        except Exception as exc:  # noqa: BLE001 - runtime boundaries report failures without losing cleanup
            self.tracer.record_error("memory", "harness.runtime", "memory_recall_failed", exc)
            return ""

    def extract_memories(self, agent) -> None:
        runtime = self.runtime_for(agent)
        if runtime.memory is not None:
            try:
                runtime.memory.extract_async(runtime.messages, agent.llm)
            except Exception as exc:  # noqa: BLE001 - runtime boundaries report failures without losing cleanup
                self.tracer.record_error("memory", "harness.runtime", "memory_extract_failed", exc)

    def set_memory_notify(self, callback) -> None:
        self._memory_notify = callback
        for runtime in self._runtimes.values():
            if runtime.memory is not None:
                runtime.memory.notify = callback

    def memory_status(self, agent):
        memory = self.runtime_for(agent).memory
        if memory is None:
            return None
        return {"directory": str(memory.store.memory_dir), "records": memory.store.list_records()}

    def skill_tool(self, agent):
        return self.runtime_for(agent).tool_by_name.get("load_skill")

    def list_skills(self, agent) -> list[dict]:
        tool = self.skill_tool(agent)
        return tool.list_skills() if tool is not None else []

    @staticmethod
    def changed_files() -> list[str]:
        from ..tools.edit import _changed_files
        return sorted(_changed_files)

    def reset_agent(self, agent) -> None:
        runtime = self.runtime_for(agent)
        self.tracer.record("context", "harness.runtime", "conversation_reset", {
            "previous_message_count": len(runtime.messages),
        })
        runtime.messages.clear()
        runtime.memory_block = ""

    def save_session(self, agent, session_id=None) -> str:
        return save_session(self.runtime_for(agent).messages, agent.llm.model, session_id)

    def resume_session(self, agent, session_id, *, restore_model=True) -> bool:
        loaded = load_session(session_id)
        if loaded is None:
            return False
        messages, model = loaded
        self.reset_agent(agent)
        self.runtime_for(agent).messages = messages
        if restore_model:
            agent.llm.model = model
        return True

    @staticmethod
    def list_sessions():
        return list_sessions()

    def answer_pending_tool_calls(self, agent, tool_calls) -> None:
        messages = self.runtime_for(agent).messages
        answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        for call in tool_calls:
            if call.id not in answered:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": "[interrupted]"})

    def run_subagent(self, parent, task: str) -> str:
        runtime = self.runtime_for(parent)
        sub = self.create_agent(
            llm=parent.llm, tools=[tool for tool in runtime.tools if tool.name != "agent"],
            max_context_tokens=runtime.context.max_tokens, max_rounds=20, memory_enabled=False,
        )
        current_input, round_count = self.state.current_input, self.state.round_count
        try:
            result = self.run_chat(sub, task)
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as exc:  # noqa: BLE001 - runtime boundaries report failures without losing cleanup
            return f"Sub-agent error: {exc}"
        finally:
            self.state.current_input, self.state.round_count = current_input, round_count
            child_runtime = self._runtimes.pop(sub)
            context_id = id(child_runtime.context)
            self._contexts.remove(context_id)
            self.state.context_ids.remove(context_id)
            self.tracer.record("context", "harness.runtime", "context_destroyed", {"context_id": context_id})

    def _close_runtime(self) -> None:
        deadline = time.monotonic() + self.config.memory_shutdown_timeout_seconds
        clients = {}
        for runtime in list(self._runtimes.values()):
            if runtime.memory is not None:
                try:
                    runtime.memory.wait_for_extraction(timeout=max(0.0, deadline - time.monotonic()))
                except Exception as exc:  # noqa: BLE001 - runtime boundaries report failures without losing cleanup
                    self.tracer.record_error("memory", "harness.runtime", "memory_shutdown_failed", exc)
            for tool in runtime.tools:
                client = getattr(tool, "client", None)
                if client is not None and callable(getattr(client, "close", None)):
                    clients[id(client)] = client
        for client in clients.values():
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - runtime boundaries report failures without losing cleanup
                self.tracer.record_error("tool", "harness.runtime", "tool_shutdown_failed", exc)

    def exec_tool(self, agent, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self.runtime_for(agent).tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        parse_error = getattr(tc, "parse_error", "")
        if parse_error:
            raw_arguments = getattr(tc, "raw_arguments", "")
            preview = raw_arguments.replace("\n", "\\n")[:300]
            return (
                f"Error: invalid JSON arguments for {tc.name}: {parse_error}. "
                f"Retry the tool call with a valid JSON object. raw_arguments_preview={preview!r}"
            )
        return self.execute_tool_call(tool, tc)

    def exec_tools(
        self, agent, tool_calls, on_tool=None, on_tool_result=None,
        *, _record_messages: bool = False,
    ) -> list[str]:
        """Parallelize read-only batches; preserve order for all other batches."""
        tool_by_name = self.runtime_for(agent).tool_by_name
        # A batch containing mutations or delegation may have dependencies.
        # Preserve model order; only known read-only batches run concurrently.
        if not all(
            tool_by_name.get(tc.name) is not None
            and tool_by_name[tc.name].read_only
            for tc in tool_calls
        ):
            results = []
            for tc in tool_calls:
                token = self.runtime_for(agent).cancel_token
                if token is not None:
                    token.raise_if_cancelled()
                if on_tool:
                    on_tool(tc.name, tc.arguments)
                result = self.exec_tool(agent, tc)
                results.append(result)
                if on_tool_result:
                    on_tool_result(tc.name, tc.arguments, result)
                if _record_messages:
                    self.runtime_for(agent).messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": result,
                    })
            return results

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # Collect results in request order, even when reads run concurrently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.exec_tool, agent, tc) for tc in tool_calls]
            results = [f.result() for f in futures]
        if on_tool_result:
            for tc, result in zip(tool_calls, results):
                on_tool_result(tc.name, tc.arguments, result)
        if _record_messages:
            for tc, result in zip(tool_calls, results):
                self.runtime_for(agent).messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
        return results

