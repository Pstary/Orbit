"""Model-facing agent; execution and state are owned by OrbitHarness."""

from .harness import OrbitHarness
from .llm import LLM
from .tools.base import Tool


class Agent:
    def __init__(self, llm: LLM, tools: list[Tool] | None = None,
                 max_context_tokens: int = 128_000, max_rounds: int = 50,
                 harness: OrbitHarness | None = None, memory_enabled: bool | None = None):
        self.llm = llm
        self.harness = harness or OrbitHarness.default()
        self.harness.register_agent(
            self, tools=tools, max_context_tokens=max_context_tokens,
            max_rounds=max_rounds, memory_enabled=memory_enabled,
        )

    def respond(self, messages, tools=None, on_token=None):
        """Make one model call; the harness decides what happens next."""
        return self.llm.chat(messages=messages, tools=tools, on_token=on_token)

    def chat(self, user_input: str, on_token=None, on_tool=None, on_tool_result=None) -> str:
        return self.harness.run_chat(self, user_input, on_token, on_tool, on_tool_result)

    def reset(self):
        self.harness.reset_agent(self)

    # Compatibility views: these do not store or manage runtime state on Agent.
    @property
    def messages(self):
        return self.harness.runtime_for(self).messages

    @messages.setter
    def messages(self, value):
        self.harness.runtime_for(self).messages = value

    @property
    def tools(self):
        return self.harness.runtime_for(self).tools

    @property
    def context(self):
        return self.harness.runtime_for(self).context

    @property
    def memory(self):
        return self.harness.runtime_for(self).memory

    @property
    def max_rounds(self):
        return self.harness.runtime_for(self).max_rounds

    @property
    def _system(self):
        return self.harness.runtime_for(self).system

    @property
    def _tool_by_name(self):
        return self.harness.runtime_for(self).tool_by_name

    def _full_messages(self):
        return self.harness.full_messages(self)

    def _tool_schemas(self):
        return self.harness.tool_schemas(self)

    def _exec_tool(self, tc):
        return self.harness.exec_tool(self, tc)

    def _exec_tools_parallel(self, tool_calls, on_tool=None, on_tool_result=None):
        return self.harness.exec_tools(self, tool_calls, on_tool, on_tool_result)

    def _answer_pending_tool_calls(self, tool_calls):
        self.harness.answer_pending_tool_calls(self, tool_calls)
