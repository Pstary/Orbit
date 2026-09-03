"""Tests for the SWE-bench Lite eval runner."""

import json
import subprocess
from pathlib import Path

from orbit.evals.swe_bench_lite import SweBenchEvalConfig, run_eval
from orbit.harness import PermissionMode
from orbit.harness.sandbox import SandboxRunner
from orbit.llm import LLMResponse, ScriptedLLM, ToolCall


def _simulate_docker(monkeypatch):
    # MEDIUM commands are required to use Docker. Simulate the container runner
    # with the local process implementation so this unit test is daemon-free.
    monkeypatch.setattr(
        SandboxRunner,
        "_run_bash_in_docker",
        lambda self, command, timeout, cancellation=None: self._run_bash_locally(
            command, timeout, cancellation,
        ),
    )


def test_swe_bench_lite_runner_with_local_jsonl(tmp_path, monkeypatch):
    _simulate_docker(monkeypatch)
    source_repo = _make_repo(tmp_path / "source")
    base_commit = _git(source_repo, "rev-parse", "HEAD").stdout.strip()
    dataset_path = tmp_path / "instances.jsonl"
    dataset_path.write_text(
        json.dumps({
            "instance_id": "local__calc-1",
            "repo": str(source_repo),
            "base_commit": base_commit,
            "problem_statement": "fix add_one so it adds one",
            "FAIL_TO_PASS": ["test_calc.py::test_add_one"],
            "PASS_TO_PASS": [],
        }) + "\n",
        encoding="utf-8",
    )

    def llm_factory(_instance, repo_dir: Path, _mode: str):
        return ScriptedLLM([
            LLMResponse(
                content="Fixing add_one.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={
                            "file_path": str(repo_dir / "calc.py"),
                            "content": "def add_one(value):\n    return value + 1\n",
                        },
                    )
                ],
            ),
            LLMResponse(content="Fixed add_one and ran the test."),
        ])

    results = run_eval(
        SweBenchEvalConfig(
            source=str(dataset_path),
            split="test",
            work_dir=tmp_path / "runs",
            force=True,
            permission_mode=PermissionMode.FULL_AUTO,
            post_test_command="python -m pytest -q",
        ),
        llm_factory,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "completed"
    assert result.passed is True
    assert result.patch_path is not None
    assert result.trace_path is not None
    assert result.result_path is not None
    assert "return value + 1" in Path(result.patch_path).read_text(encoding="utf-8")
    assert Path(result.trace_path).exists()
    assert Path(result.result_path).exists()
    assert list((tmp_path / "runs" / "local__calc-1" / "test_logs").glob("test-run-*.json"))


def test_swe_bench_lite_runner_compares_agent_and_direct_modes(tmp_path, monkeypatch):
    _simulate_docker(monkeypatch)
    source_repo = _make_repo(tmp_path / "source")
    base_commit = _git(source_repo, "rev-parse", "HEAD").stdout.strip()
    dataset_path = tmp_path / "instances.jsonl"
    dataset_path.write_text(
        json.dumps({
            "instance_id": "local__calc-compare",
            "repo": str(source_repo),
            "base_commit": base_commit,
            "problem_statement": "fix add_one so it adds one",
            "FAIL_TO_PASS": ["test_calc.py::test_add_one"],
            "PASS_TO_PASS": [],
        }) + "\n",
        encoding="utf-8",
    )

    def llm_factory(_instance, repo_dir: Path, mode: str):
        if mode == "direct":
            return ScriptedLLM([
                LLMResponse(content=(
                    "diff --git a/calc.py b/calc.py\n"
                    "index 372d88e..49ee62c 100644\n"
                    "--- a/calc.py\n"
                    "+++ b/calc.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def add_one(value):\n"
                    "-    return value\n"
                    "+    return value + 1\n"
                )),
            ])
        return ScriptedLLM([
            LLMResponse(
                content="Fixing add_one.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={
                            "file_path": str(repo_dir / "calc.py"),
                            "content": "def add_one(value):\n    return value + 1\n",
                        },
                    )
                ],
            ),
            LLMResponse(content="Fixed add_one and ran the test."),
        ])

    run_dir = tmp_path / "runs"
    results = run_eval(
        SweBenchEvalConfig(
            source=str(dataset_path),
            split="test",
            eval_mode="both",
            work_dir=run_dir,
            force=True,
            permission_mode=PermissionMode.FULL_AUTO,
            post_test_command="python -m pytest -q",
        ),
        llm_factory,
    )

    assert {result.mode for result in results} == {"agent", "direct"}
    assert all(result.passed is True for result in results)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["by_mode"]["agent"]["success_rate"] == 1.0
    assert summary["by_mode"]["direct"]["success_rate"] == 1.0
    assert (run_dir / "agent" / "local__calc-compare" / "model.patch").exists()
    assert (run_dir / "direct" / "local__calc-compare" / "model.patch").exists()


def _make_repo(path: Path) -> Path:
    path.mkdir()
    (path / "calc.py").write_text("def add_one(value):\n    return value\n", encoding="utf-8")
    (path / "test_calc.py").write_text(
        "from calc import add_one\n\n"
        "def test_add_one():\n"
        "    assert add_one(1) == 2\n",
        encoding="utf-8",
    )
    _git(path, "init")
    _git(path, "add", "calc.py", "test_calc.py")
    _git(path, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "base")
    return path


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
