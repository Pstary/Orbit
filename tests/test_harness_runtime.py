"""Ownership and lifecycle guarantees of the harness-driven agent."""

from types import SimpleNamespace

import pytest

from orbit.agent import Agent
from orbit.harness import HarnessConfig, OrbitHarness, PermissionMode
from orbit.llm import LLMResponse, ScriptedLLM, ToolCall
from orbit.tools.agent import AgentTool
from orbit.tools.read import ReadFileTool


def make_harness(tmp_path, **kwargs):
    return OrbitHarness(HarnessConfig(
        workspace_root=tmp_path, trace_dir=tmp_path / "traces",
        permission_mode=PermissionMode.FULL_AUTO, mcp_enabled=False,
        **kwargs,
    ))


def test_agent_only_owns_model_and_harness(tmp_path):
    harness = make_harness(tmp_path)
    agent = Agent(ScriptedLLM([]), tools=[], harness=harness)
    runtime = harness.runtime_for(agent)
    assert set(vars(agent)) == {"llm", "harness"}
    assert agent.messages is runtime.messages
    assert agent.context is runtime.context
    assert agent.memory is runtime.memory
    assert str(tmp_path) in runtime.system


@pytest.mark.parametrize("direct", [True, False])
def test_both_entry_points_run_memory_and_lifecycle_once(tmp_path, direct):
    harness = make_harness(tmp_path)
    agent = harness.create_agent(ScriptedLLM([LLMResponse(content="done")]), tools=[])
    calls = []
    runtime = harness.runtime_for(agent)
    runtime.memory = SimpleNamespace(
        recall_block=lambda messages, llm: calls.append("recall") or "remember this",
        extract_async=lambda messages, llm: calls.append("extract"),
        wait_for_extraction=lambda **kwargs: calls.append("wait"),
    )
    result = agent.chat("hello") if direct else harness.run_chat(agent, "hello")
    assert result == "done"
    assert calls == ["recall", "extract"]
    assert "remember this" in harness.full_messages(agent)[0]["content"]
    actions = [event.action for event in harness.tracer.events]
    assert actions.count("execution_started") == actions.count("execution_finished") == 1
    assert actions.count("chat_started") == actions.count("chat_finished") == 1
    harness.close()
    harness.close()
    assert calls == ["recall", "extract", "wait"]


def test_reset_and_resume_are_owned_by_harness(tmp_path, monkeypatch):
    from orbit import session

    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "sessions")
    harness = make_harness(tmp_path)
    agent = harness.create_agent(ScriptedLLM([LLMResponse(content="done")]), tools=[], memory_enabled=False)
    agent.chat("hello")
    sid = harness.save_session(agent)
    runtime = harness.runtime_for(agent)
    runtime.memory_block = "stale"
    agent.reset()
    assert not runtime.messages and not runtime.memory_block
    assert harness.resume_session(agent, sid)
    assert runtime.messages[0]["content"] == "hello"
    assert not runtime.memory_block


def test_agents_sharing_harness_have_separate_state_and_bindings(tmp_path):
    harness = make_harness(tmp_path)
    shared = AgentTool()
    first = harness.create_agent(ScriptedLLM([]), tools=[shared])
    second = harness.create_agent(ScriptedLLM([]), tools=[shared], memory_enabled=False)
    first.messages.append({"role": "user", "content": "first only"})
    assert second.messages == []
    assert first.context is not second.context
    assert first.memory is not None and second.memory is None
    assert first.tools[0]._parent_agent is first
    assert second.tools[0]._parent_agent is second
    assert first.tools[0] is not shared


def test_subagent_runs_through_harness_and_releases_context(tmp_path):
    harness = make_harness(tmp_path)
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("sub", "agent", {"task": "look around"})]),
        LLMResponse(content="child report"),
        LLMResponse(content="parent done"),
    ])
    parent = harness.create_agent(llm, tools=[AgentTool()], memory_enabled=False)
    assert parent.chat("delegate") == "parent done"
    assert "child report" in parent.messages[2]["content"]
    assert list(harness._runtimes) == [parent]
    assert harness.state.context_ids == [id(parent.context)]
    assert harness.state.current_input == "delegate"


def test_harness_close_waits_memory_then_closes_shared_client_once(tmp_path, monkeypatch):
    harness = make_harness(tmp_path)
    events = []
    tool = ReadFileTool()
    tool.client = SimpleNamespace(close=lambda: events.append("client"))
    first = harness.create_agent(ScriptedLLM([]), tools=[tool])
    harness.create_agent(ScriptedLLM([]), tools=[tool], memory_enabled=False)
    harness.runtime_for(first).memory = SimpleNamespace(
        wait_for_extraction=lambda **kwargs: events.append("memory"),
    )
    monkeypatch.setattr(harness, "save_trace", lambda: events.append("trace"))
    harness.close()
    assert events == ["memory", "client", "trace"]
    harness.close()
    assert events.count("client") == events.count("memory") == 1
    with pytest.raises(RuntimeError, match="closed"):
        first.chat("after close")


def test_memory_failure_does_not_abort_chat(tmp_path):
    harness = make_harness(tmp_path)
    agent = harness.create_agent(ScriptedLLM([LLMResponse(content="done")]), tools=[])

    def fail(*args, **kwargs):
        raise RuntimeError("memory unavailable")

    harness.runtime_for(agent).memory = SimpleNamespace(recall_block=fail, extract_async=fail)
    assert agent.chat("hello") == "done"
    errors = [event.action for event in harness.tracer.events]
    assert "memory_recall_failed" in errors
    assert "memory_extract_failed" in errors


def test_callback_failure_repairs_tool_history(tmp_path):
    harness = make_harness(tmp_path)
    agent = harness.create_agent(ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "absent"})]),
        LLMResponse(content="recovered"),
    ]), tools=[ReadFileTool()], memory_enabled=False)

    def fail(*args):
        raise RuntimeError("display failure")

    with pytest.raises(RuntimeError, match="display failure"):
        agent.chat("read", on_tool_result=fail)
    assert agent.messages[-1] == {"role": "tool", "tool_call_id": "read", "content": "[interrupted]"}
    assert agent.chat("continue") == "recovered"


def test_context_manager_closes_on_exception(tmp_path):
    harness = make_harness(tmp_path)
    with pytest.raises(ValueError), harness:
        raise ValueError("failure")
    assert harness.state.closed
    assert harness.tracer.trace_path.exists()


def test_config_disables_memory_and_skills_without_cli(tmp_path):
    harness = make_harness(tmp_path, memory_enabled=False, skills_enabled=False)
    agent = harness.create_agent(ScriptedLLM([]))
    assert agent.memory is None
    assert harness.skill_tool(agent) is None
