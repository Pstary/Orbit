"""Sub-agent spawning (inspired by Claude Code's AgentTool, 1397 lines).

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

The sub-agent runs to completion and returns a text summary.
"""

from typing import ClassVar

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None

    def execute(self, task: str) -> str:
        # AgentTool必须挂到父Agent上才能工作，否则无法复用父Agent的LLM和工具列表。
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # 延迟导入Agent，避免模块加载阶段出现循环依赖。
        from ..agent import Agent

        parent = self._parent_agent
        # 创建一个独立子Agent：复用父Agent的LLM和上下文上限，但拥有自己的对话历史。
        sub = Agent(
            llm=parent.llm,
            # 过滤掉agent工具，禁止子Agent继续创建子Agent导致递归失控。
            tools=[t for t in parent.tools if t.name != "agent"],
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
            harness=parent.harness,
            # 子Agent是一次性调研/执行上下文，不召回也不写入长期记忆，避免额外LLM调用和记忆污染。
            memory_enabled=False,
        )

        # 子Agent失败只返回文本错误，不把异常继续抛给父Agent。
        try:
            result = sub.chat(task)
            # 子Agent结果会进入父Agent上下文，过长时需要截断以控制上下文膨胀。
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:  # noqa: BLE001
            return f"Sub-agent error: {e}"
