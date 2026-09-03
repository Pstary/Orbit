"""Regression tests for workspace governance and execution lifecycle."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from orbit.agent import Agent
from orbit.cancellation import ToolInterrupted
from orbit.harness import HarnessConfig, OrbitHarness, PermissionMode
from orbit.llm import LLMResponse, ScriptedLLM, ToolCall
from orbit.mcp import McpServerConfig, McpStdioClient
from orbit.memory import MemoryManager
from orbit.tools import get_tool
from orbit.tools.base import Tool
from orbit.tools.runtime import check_deadline


def harness_at(root, **kwargs):
    return OrbitHarness(HarnessConfig(
        workspace_root=root, trace_dir=root / "traces",
        permission_mode=PermissionMode.FULL_AUTO, **kwargs,
    ))


def call(harness, name, **arguments):
    return harness.execute_tool_call(get_tool(name), ToolCall("test", name, arguments))


def test_relative_paths_and_search_defaults_use_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    harness = harness_at(workspace)
    assert "Wrote" in call(harness, "write_file", file_path="hello.txt", content="before")
    assert not (other / "hello.txt").exists()
    assert "Edited" in call(harness, "edit_file", file_path="hello.txt", old_string="before", new_string="after")
    assert "after" in call(harness, "read_file", file_path="hello.txt")
    assert "after" in call(harness, "grep", pattern="after")
    assert "hello.txt" in call(harness, "glob", pattern="*.txt")


def test_search_checks_each_file_and_glob_cannot_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".env").write_text("MARKER_PRIVATE=fake", encoding="utf-8")
    (root / "public.txt").write_text("MARKER_PUBLIC=ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("MARKER_OUTSIDE=fake", encoding="utf-8")
    harness = harness_at(root)
    result = call(harness, "grep", pattern="MARKER", path=".")
    assert "MARKER_PUBLIC" in result
    assert "MARKER_PRIVATE" not in result
    assert ".env" not in call(harness, "glob", pattern="*")
    assert "outside.txt" not in call(harness, "glob", pattern="../*.txt")
    assert "Blocked" in call(harness, "read_file", file_path=".env")


def test_search_skips_symlink_to_outside_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET_MARKER", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows requires symlink privileges")
        raise
    result = call(harness_at(root), "grep", pattern="SECRET_MARKER")
    assert "SECRET_MARKER" not in result


def test_plan_cannot_approve_protected_write(tmp_path):
    approvals = []
    harness = OrbitHarness(
        HarnessConfig(workspace_root=tmp_path, trace_dir=tmp_path / "traces", permission_mode=PermissionMode.PLAN),
        approval_callback=lambda *args: approvals.append(args) or True,
    )
    harness.policy.settings.allowed_tools.add("write_file")
    result = call(harness, "write_file", file_path="orbit/agent.py", content="bad")
    assert "plan mode" in result
    assert approvals == []
    assert not (tmp_path / "orbit" / "agent.py").exists()


class SlowMutation(Tool):
    name = "slow_mutation"
    description = "Test mutation"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, path, cooperative=False):
        self.path = path
        self.cooperative = cooperative
        self.calls = 0
        self.finished = threading.Event()

    def execute(self):
        self.calls += 1
        try:
            time.sleep(0.08)
            if self.cooperative:
                return get_tool("write_file").execute(str(self.path), "late")
            self.path.write_text("completed before timeout returned", encoding="utf-8")
            return "done"
        finally:
            self.finished.set()


@pytest.mark.parametrize("cooperative", [False, True])
def test_timeout_settles_worker_without_retries(tmp_path, cooperative):
    harness = harness_at(tmp_path, tool_timeout_seconds=0.02, max_retries=3)
    target = tmp_path / "late.txt"
    tool = SlowMutation(target, cooperative=cooperative)
    result = harness.execute_tool_call(tool, ToolCall("slow", tool.name, {}))
    assert "timed out" in result
    assert tool.finished.is_set()  # No running tool is left behind on return.
    assert tool.calls == 1
    assert target.exists() is not cooperative


def test_interrupt_repairs_history_and_next_prompt_continues(tmp_path):
    started = threading.Event()

    class CancellableTool(Tool):
        name = "cancellable"
        description = "Wait until interrupted"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self):
            started.set()
            while True:
                check_deadline()
                time.sleep(0.01)

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("wait", "cancellable", {})]),
        LLMResponse(content="continued with the new instruction"),
    ])
    harness = harness_at(tmp_path, memory_enabled=False)
    agent = Agent(llm, tools=[CancellableTool()], harness=harness, memory_enabled=False)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.chat, "start the task")
        assert started.wait(timeout=2)
        assert harness.interrupt(agent)
        with pytest.raises(ToolInterrupted):
            future.result(timeout=2)

    assert agent.messages[-1] == {
        "role": "tool", "tool_call_id": "wait", "content": "[interrupted]",
    }
    assert agent.chat("use the smaller test scope") == "continued with the new instruction"
    assert agent.messages[-2] == {"role": "user", "content": "use the smaller test scope"}


def test_interrupt_keeps_results_from_earlier_tools_in_batch(tmp_path):
    started = threading.Event()

    class First(Tool):
        name = "first"
        description = "Complete immediately"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self):
            return "completed"

    class Second(Tool):
        name = "second"
        description = "Wait until interrupted"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self):
            started.set()
            while True:
                check_deadline()
                time.sleep(0.01)

    harness = harness_at(tmp_path, memory_enabled=False)
    agent = Agent(
        ScriptedLLM([LLMResponse(tool_calls=[
            ToolCall("one", "first", {}), ToolCall("two", "second", {}),
        ])]),
        tools=[First(), Second()], harness=harness, memory_enabled=False,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.chat, "start")
        assert started.wait(timeout=2)
        assert harness.interrupt(agent)
        with pytest.raises(ToolInterrupted):
            future.result(timeout=2)

    replies = {m["tool_call_id"]: m["content"] for m in agent.messages if m.get("role") == "tool"}
    assert replies == {"one": "completed", "two": "[interrupted]"}


def test_local_bash_can_be_interrupted(tmp_path):
    harness = harness_at(tmp_path, tool_timeout_seconds=30, memory_enabled=False)
    agent = Agent(
        ScriptedLLM([LLMResponse(tool_calls=[ToolCall(
            "sleep", "bash", {"command": f'"{sys.executable}" -c "import time; time.sleep(30)"'},
        )])]),
        tools=[get_tool("bash")], harness=harness, memory_enabled=False,
    )

    started_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.chat, "run a long command")
        deadline = time.monotonic() + 2
        while harness.runtime_for(agent).cancel_token is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert harness.interrupt(agent)
        with pytest.raises(ToolInterrupted):
            future.result(timeout=5)
    assert time.monotonic() - started_at < 5


def test_mutation_batch_preserves_write_edit_read_order(tmp_path):
    harness = harness_at(tmp_path)
    agent = Agent(ScriptedLLM([]), tools=[get_tool(n) for n in ("write_file", "edit_file", "read_file")],
                  harness=harness, memory_enabled=False)
    calls = [
        ToolCall("1", "write_file", {"file_path": "x.txt", "content": "old"}),
        ToolCall("2", "edit_file", {"file_path": "x.txt", "old_string": "old", "new_string": "new"}),
        ToolCall("3", "read_file", {"file_path": "x.txt"}),
    ]
    results = agent._exec_tools_parallel(calls)
    assert "Edited" in results[1]
    assert "new" in results[2]


def test_concurrent_file_edits_do_not_lose_changes(tmp_path, monkeypatch):
    target = tmp_path / "x.txt"
    target.write_text("first second", encoding="utf-8")
    original_read = Path.read_text

    def slow_read(path, *args, **kwargs):
        content = original_read(path, *args, **kwargs)
        if path == target:
            time.sleep(0.05)
        return content

    monkeypatch.setattr(Path, "read_text", slow_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(get_tool("edit_file").execute, str(target), old, new)
                   for old, new in (("first", "FIRST"), ("second", "SECOND"))]
        assert all("Edited" in f.result() for f in futures)
    assert original_read(target, encoding="utf-8") == "FIRST SECOND"


def test_read_only_batch_still_runs_concurrently(tmp_path):
    barrier = threading.Barrier(2, timeout=2)

    class ReadOnly(Tool):
        name = "read_only"
        description = "Parallel read test"
        read_only = True
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self):
            barrier.wait()
            return "ok"

    agent = Agent(ScriptedLLM([]), tools=[ReadOnly()], harness=harness_at(tmp_path), memory_enabled=False)
    results = agent._exec_tools_parallel([ToolCall("1", "read_only", {}), ToolCall("2", "read_only", {})])
    assert results == ["ok", "ok"]


@pytest.mark.parametrize("prefix,reader", [
    (b"", "line"), (b"partial line", "line"), (b"abc", "body"),
])
def test_mcp_partial_or_silent_stdout_obeys_deadline(prefix, reader):
    code = f"import sys,time; sys.stdout.buffer.write({prefix!r}); sys.stdout.flush(); time.sleep(10)"
    client = McpStdioClient(McpServerConfig("test", sys.executable, ["-u", "-c", code]))
    client._start_process()
    proc = client._process
    try:
        with pytest.raises(TimeoutError):
            deadline = time.monotonic() + 0.15
            if reader == "line":
                client._readline(deadline)
            else:
                client._read_exact(10, deadline)
    finally:
        client.close()
    assert proc.poll() is not None


def test_mcp_reader_preserves_buffered_utf8_frames():
    payload = '{"result":"中文"}'.encode()
    data = f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload + b"next\n"
    code = f"import sys,time; sys.stdout.buffer.write({data!r}); sys.stdout.flush(); time.sleep(10)"
    client = McpStdioClient(McpServerConfig("test", sys.executable, ["-u", "-c", code]))
    client._start_process()
    try:
        deadline = time.monotonic() + 2
        assert client._read_message(deadline) == {"result": "中文"}
        assert client._readline(deadline) == "next\n"
    finally:
        client.close()


def test_memory_shutdown_waits_for_pending_write(tmp_path, monkeypatch):
    manager = MemoryManager(tmp_path)
    finished = threading.Event()

    def extract(messages, llm):
        time.sleep(0.05)
        (tmp_path / "saved.txt").write_text("saved", encoding="utf-8")
        finished.set()

    monkeypatch.setattr(manager, "_run_extraction_safely", extract)
    manager.extract_async([{"role": "user", "content": "remember"}], object())
    assert manager.wait_for_extraction(timeout=2)
    assert finished.is_set()
    assert (tmp_path / "saved.txt").exists()


def test_memory_shutdown_timeout_is_reported(tmp_path, monkeypatch):
    messages = []
    manager = MemoryManager(tmp_path, notify=messages.append)
    release = threading.Event()
    monkeypatch.setattr(manager, "_run_extraction_safely", lambda *args: release.wait(2))
    manager.extract_async([{"role": "user", "content": "remember"}], object())
    try:
        assert not manager.wait_for_extraction(timeout=0.01)
        assert messages
    finally:
        release.set()
        assert manager.wait_for_extraction(timeout=2)


@pytest.mark.parametrize("one_shot", [False, True])
def test_cli_waits_for_memory_before_saving_trace(tmp_path, monkeypatch, one_shot):
    from orbit import cli
    from orbit.config import Config

    events = []
    harness = harness_at(tmp_path)
    agent = harness.create_agent(llm=ScriptedLLM([]), tools=[])
    harness.runtime_for(agent).memory = SimpleNamespace(
        wait_for_extraction=lambda **kwargs: events.append("memory_finished"),
    )
    monkeypatch.setattr(harness, "create_agent", lambda **kwargs: agent)
    monkeypatch.setattr(harness, "save_trace", lambda: events.append("trace_saved"))
    monkeypatch.setattr(sys, "argv", ["orbit", "-p", "hello"] if one_shot else ["orbit"])
    monkeypatch.setattr(cli, "parse_config", lambda: Config(api_key="fake"))
    monkeypatch.setattr(cli, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(cli, "_build_harness", lambda config: harness)
    monkeypatch.setattr(cli, "_run_once", lambda *args: events.append("chat"))
    monkeypatch.setattr(cli, "_repl", lambda *args: events.append("chat"))
    cli.main()
    assert events == ["chat", "memory_finished", "trace_saved"]
