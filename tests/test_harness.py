"""Tests for the CoreCoder harness layer."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from corecoder.harness import CoreCoderHarness, HarnessConfig, HookEvent, HookResult, PermissionMode
from corecoder.harness.sandbox import default_test_log_dir
from corecoder.harness.trace import default_trace_dir
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.tools import get_tool


@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: dict


def test_harness_wraps_chat_and_saves_trace(tmp_path):
    harness = CoreCoderHarness(HarnessConfig(
        workspace_root=tmp_path,
        trace_dir=tmp_path / "traces",
        permission_mode=PermissionMode.FULL_AUTO,
    ))
    target = tmp_path / "hello.txt"
    llm = ScriptedLLM([
        LLMResponse(
            content="writing",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="write_file",
                    arguments={"file_path": str(target), "content": "hello\n"},
                )
            ],
        ),
        LLMResponse(content="done"),
    ])
    agent = harness.create_agent(llm=llm, max_rounds=4)

    assert harness.run_chat(agent, "create a file") == "done"
    assert not harness.tracer.trace_path.exists()
    trace_path = harness.close()

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    actions = [event["action"] for event in payload["events"]]
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert "harness_created" in actions
    assert "context_created" in actions
    assert "context_updated" in actions
    assert "context_destroyed" in actions
    assert "llm_call_started" in actions
    assert "permission_checked" in actions
    assert "tool_execution_finished" in actions
    assert "chat_finished" in actions


def test_harness_blocks_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-corecoder-harness.txt"
    outside.write_text("secret", encoding="utf-8")
    harness = CoreCoderHarness(HarnessConfig(
        workspace_root=tmp_path,
        trace_dir=tmp_path / "traces",
        permission_mode=PermissionMode.FULL_AUTO,
    ))
    read = get_tool("read_file")
    result = harness.execute_tool_call(
        read,
        _ToolCall(id="c1", name="read_file", arguments={"file_path": str(outside)}),
    )

    assert "Blocked by harness" in result
    assert "outside workspace boundary" in result


def test_default_mode_requires_approval_for_write(tmp_path):
    harness = CoreCoderHarness(HarnessConfig(
        workspace_root=tmp_path,
        trace_dir=tmp_path / "traces",
        permission_mode=PermissionMode.DEFAULT,
    ))
    write = get_tool("write_file")
    target = tmp_path / "blocked.txt"
    result = harness.execute_tool_call(
        write,
        _ToolCall(
            id="c1",
            name="write_file",
            arguments={"file_path": str(target), "content": "blocked"},
        ),
    )

    assert "Blocked by harness approval" in result
    assert not target.exists()


def test_custom_pre_tool_hook_can_block_execution(tmp_path):
    harness = CoreCoderHarness(HarnessConfig(
        workspace_root=tmp_path,
        trace_dir=tmp_path / "traces",
        permission_mode=PermissionMode.FULL_AUTO,
    ))

    def block_bash(payload):
        if payload["tool_name"] == "bash":
            return HookResult(blocked=True, reason="bash disabled by test hook")
        return None

    harness.hooks.register(HookEvent.PRE_TOOL_USE, block_bash)
    bash = get_tool("bash")
    result = harness.execute_tool_call(
        bash,
        _ToolCall(id="c1", name="bash", arguments={"command": "echo should-not-run"}),
    )

    assert result == "Blocked by harness: bash disabled by test hook"


def test_default_trace_dir_uses_package_date_directory():
    date_dir = datetime.now(timezone.utc).strftime("%Y%m%d")
    trace_dir = default_trace_dir()

    assert trace_dir.parts[-3:] == ("corecoder", "trace", date_dir)


def test_default_test_log_dir_uses_project_tests_date_directory():
    date_dir = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_dir = default_test_log_dir()

    assert log_dir.parts[-3:] == ("tests", "logs", date_dir)


def test_bash_test_command_writes_test_log(tmp_path):
    test_log_dir = tmp_path / "tests" / "logs" / "20990101"
    harness = CoreCoderHarness(HarnessConfig(
        workspace_root=tmp_path,
        trace_dir=tmp_path / "traces",
        test_log_dir=test_log_dir,
        permission_mode=PermissionMode.FULL_AUTO,
    ))
    bash = get_tool("bash")

    result = harness.execute_tool_call(
        bash,
        _ToolCall(id="c1", name="bash", arguments={"command": "python -m pytest --version"}),
    )
    trace_path = harness.close()

    logs = list(test_log_dir.glob("test-run-*.json"))
    assert logs
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["command"] == "python -m pytest --version"
    assert payload["returncode"] == 0
    assert "pytest" in result.lower()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "test_log_saved" in [event["action"] for event in trace["events"]]
