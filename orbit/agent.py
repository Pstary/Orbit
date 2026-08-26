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
from .prompt import system_prompt
from .tools import ALL_TOOLS
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
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.harness = harness or OrbitHarness.default()
        self.context = self.harness.create_context(max_tokens=max_context_tokens)
        self.harness.attach_context(self.context)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.harness.tracer.record("agent", "agent", "agent_initialized", {
            "tool_names": [t.name for t in self.tools],
            "max_rounds": self.max_rounds,
            "max_context_tokens": max_context_tokens,
        })

        # Agent初始化时会遍历所有工具。如果某个工具是 AgentTool ，就把当前这个 Agent对象塞进工具的_parent_agent字段里。
        # 这是子Agent能力的 wiring/binding代码，用来让agent工具拿到父 Agent上下文，从而派生一个受控的子Agent。
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    # 把 Agent内部的工具对象列表，转换成可以传给大模型的 tools schema列表。
    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.harness.start_chat(user_input)
        self.messages.append({"role": "user", "content": user_input})
        # 在把完整历史发给模型前，先做上下文治理，避免 messages 太长导致超过模型上下文窗口。
        self.harness.compress_context_if_needed(self.context, self.messages, self.llm)

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
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                self.harness.finish_chat(resp.content, reason="assistant_final")
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
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
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
        return result

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        return self.harness.execute_tool_call(tool, tc)

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
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
            return [f.result() for f in futures]

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
