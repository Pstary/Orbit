"""Core agent loop.

This is the heart of Orbit.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures

from .harness import OrbitHarness
from .llm import LLM
from .memory import MemoryManager
from .prompt import system_prompt
from .tools import get_default_tools
from .tools.agent import AgentTool
from .tools.base import Tool

# agent.py实现的是Orbit的最小Agent执行引擎，核心是一个受max_rounds限制的ReAct-style循环：模型决定工具调用，Agent执行工具并回填结果，直到模型不再调用工具并返回最终回答。
class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        harness: OrbitHarness | None = None,
        memory_enabled: bool = True,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else get_default_tools()
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.harness = harness or OrbitHarness.default()
        self.context = self.harness.create_context(max_tokens=max_context_tokens)
        self.harness.attach_context(self.context)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        # 跨会话记忆：.memory/ 目录里的持久知识。每轮对话前召回相关记忆注入system prompt，
        # 对话结束后从对话中提取值得长期保留的用户偏好/项目事实。best-effort，失败不影响主流程。
        self.memory = (
            MemoryManager(
                self.harness.workspace_root,
                tracer=self.harness.tracer,
                memory_dir=getattr(self.harness, "memory_dir", None),
            )
            if memory_enabled
            else None
        )
        # 当前轮召回的记忆块，由_full_messages()拼进system消息；不写入messages历史。
        self._memory_block = ""
        self.harness.tracer.record("agent", "agent", "agent_initialized", {
            "tool_names": [t.name for t in self.tools],
            "max_rounds": self.max_rounds,
            "max_context_tokens": max_context_tokens,
            "model": getattr(self.llm, "model", None),
            "memory_enabled": memory_enabled,
        })

        # Agent初始化时会遍历所有工具。如果某个工具是 AgentTool ，就把当前这个 Agent对象塞进工具的_parent_agent字段里。
        # 这是子Agent能力的 wiring/binding代码，用来让agent工具拿到父 Agent上下文，从而派生一个受控的子Agent。
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        # 记忆块每轮动态生成（可能新增记忆），拼在静态system prompt之后，不污染messages历史。
        system_content = self._system
        if self._memory_block:
            system_content = f"{system_content}\n\n{self._memory_block}"
        return [{"role": "system", "content": system_content}] + self.messages

    # 把 Agent内部的工具对象列表，转换成可以传给大模型的 tools schema列表。
    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None, on_tool_result=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.harness.start_chat(user_input)
        self.messages.append({"role": "user", "content": user_input})
        # 在把完整历史发给模型前，先做上下文治理，避免 messages 太长导致超过模型上下文窗口。
        self.harness.compress_context_if_needed(self.context, self.messages, self.llm)
        # 召回与本轮请求相关的持久记忆，拼进system prompt（best-effort，离线脚本模型自动跳过）。
        self._memory_block = self._recall_memories()

        for round_index in range(1, self.max_rounds + 1):
            self.harness.record_round_start(round_index, self.max_rounds, len(self.messages))
            resp = self.harness.run_llm_call(
                lambda: self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    # on_token=on_token 是在把流式输出处理函数透传给模型层，让模型每生成一段内容就能实时显示到终端。
                    on_token=on_token,
                ),
                round_index=round_index,
                # 传入 llm 供 harness 记录 model 名、单次/累计 token 与费用。
                llm=self.llm,
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                self.harness.finish_chat(resp.content, reason="assistant_final")
                # 对话结束：从本轮对话提取持久记忆（best-effort，失败静默）。
                self._extract_memories()
                return resp.content

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    # 当模型决定调用某个工具时，先通知外层“模型要调用工具了”，然后真正执行工具，并把工具结果写回对话历史。
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    if on_tool_result:
                        on_tool_result(tc.name, tc.arguments, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool, on_tool_result)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.harness.compress_context_if_needed(self.context, self.messages, self.llm)

        result = "(reached maximum tool-call rounds)"
        self.harness.finish_chat(result, reason="max_rounds")
        self._extract_memories()
        return result

    def _recall_memories(self) -> str:
        """召回相关记忆；记忆关闭或任何异常都返回空串，绝不影响主循环。"""
        if self.memory is None:
            return ""
        try:
            return self.memory.recall_block(self.messages, self.llm)
        except Exception as exc:  # noqa: BLE001
            self.harness.tracer.record_error("memory", "agent", "memory_recall_failed", exc)
            return ""

    def _extract_memories(self) -> None:
        """对话结束后触发持久记忆提取；best-effort，异常静默。

        触发时机（对齐 Claude Code）：本方法只在 LLM 本轮【没有 tool_calls】
        时被调用——说明模型不再需要工具，上一个任务刚结束，正是提取持久记忆
        的时机。提取本身是 fire-and-forget：spawn 一个后台 forked agent 异步
        执行（独立对话、最多 5 轮、只能读文件/写记忆文件、不写主 trace），
        不阻塞主 Agent 把回复返回给用户。
        """
        if self.memory is None:
            return
        try:
            self.memory.extract_async(self.messages, self.llm)
        except Exception as exc:  # noqa: BLE001
            self.harness.tracer.record_error("memory", "agent", "memory_extract_failed", exc)

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
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
        return self.harness.execute_tool_call(tool, tc)

    def _exec_tools_parallel(self, tool_calls, on_tool=None, on_tool_result=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # 这是 Agent的多工具并发执行逻辑，用线程池最多同时跑8个工具，然后按原顺序收集工具结果。
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            results = [f.result() for f in futures]
        if on_tool_result:
            for tc, result in zip(tool_calls, results):
                on_tool_result(tc.name, tc.arguments, result)
        return results

    # 处理工具调用被中断后的消息补全问题。
    # 当模型在执行过程中被中断了，比如用户按下了Ctrl+C，那么模型会返回一个tool_calls消息，但是没有对应的tool reply消息。
    # 我们需要在下一次请求时，把这些tool_calls消息对应的tool reply消息也补全到对话历史里
    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        # 先找出历史里已经回答过的工具调用ID
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.harness.tracer.record("context", "agent", "conversation_reset", {
            "previous_message_count": len(self.messages),
        })
        self.messages.clear()
        # 清空对话的同时重置本轮记忆块，避免旧召回内容残留到下一轮。
        self._memory_block = ""
