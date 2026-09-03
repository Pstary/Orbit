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

    # Bound by the harness when registering an agent.
    _parent_agent = None

    def execute(self, task: str) -> str:
        # AgentTool必须挂到父Agent上才能工作，否则无法复用父Agent的LLM和工具列表。
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        return self._parent_agent.harness.run_subagent(self._parent_agent, task)
