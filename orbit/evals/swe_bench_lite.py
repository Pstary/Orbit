"""SWE-bench Lite runner for evaluating Orbit as a coding agent.

This runner is intentionally lightweight. It executes Orbit against each
SWE-bench Lite instance, stores the produced patch, saves the full harness
trace, and optionally runs a verification command. It does not replace the
official SWE-bench grader.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from orbit.config import Config, ConfigError, parse_config
from orbit.harness import HarnessConfig, OrbitHarness, PermissionMode
from orbit.llm import LLM, LiteLLM

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"


def _date_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def default_eval_run_dir() -> Path:
    return Path(__file__).resolve().parent / "runs" / _date_dir()


@dataclass
class SweBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweBenchEvalConfig:
    source: str = DEFAULT_DATASET
    split: str = "test"
    limit: int | None = None
    instance_ids: list[str] = field(default_factory=list)
    eval_mode: str = "agent"
    work_dir: Path = field(default_factory=default_eval_run_dir)
    repo_cache_dir: Path | None = None
    keep_workspaces: bool = True
    force: bool = False
    permission_mode: PermissionMode = PermissionMode.FULL_AUTO
    sandbox_backend: str = "local"
    docker_image: str = "python:3.13-slim"
    tool_timeout_seconds: int = 600
    max_retries: int = 0
    max_rounds: int = 30
    max_context_tokens: int = 128_000
    apply_test_patch: bool = False
    post_test_command: str = ""


@dataclass
class SweBenchEvalResult:
    mode: str
    instance_id: str
    repo: str
    base_commit: str
    status: str
    passed: bool | None
    duration_ms: float
    workspace_path: str
    patch_path: str | None = None
    trace_path: str | None = None
    result_path: str | None = None
    response: str = ""
    test_command: str = ""
    test_output: str = ""
    test_log_path: str | None = None
    error: str | None = None
    stack: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweBenchEvalSummary:
    total: int
    by_mode: dict[str, dict[str, Any]]


LLMFactory = Callable[[SweBenchInstance, Path, str], Any]


def load_instances(source: str, split: str, limit: int | None = None, instance_ids: Iterable[str] = ()) -> list[SweBenchInstance]:
    wanted = set(instance_ids)
    rows = list(_load_rows(source, split))
    instances: list[SweBenchInstance] = []
    for row in rows:
        instance = _normalize_instance(row)
        if wanted and instance.instance_id not in wanted:
            continue
        instances.append(instance)
        if limit is not None and len(instances) >= limit:
            break
    return instances


def _eval_modes(eval_mode: str) -> list[str]:
    if eval_mode == "both":
        return ["agent", "direct"]
    if eval_mode in {"agent", "direct"}:
        return [eval_mode]
    raise ValueError(f"unsupported eval mode: {eval_mode}")


def run_eval(config: SweBenchEvalConfig, llm_factory: LLMFactory) -> list[SweBenchEvalResult]:
    config.work_dir = Path(config.work_dir).expanduser().resolve()
    config.work_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.work_dir / "results.jsonl"
    instances = load_instances(config.source, config.split, config.limit, config.instance_ids)
    modes = _eval_modes(config.eval_mode)
    results: list[SweBenchEvalResult] = []
    for mode in modes:
        for index, instance in enumerate(instances, start=1):
            result = run_instance(
                instance,
                config,
                llm_factory,
                mode=mode,
                isolate_mode_dir=len(modes) > 1,
                index=index,
                total=len(instances),
            )
            results.append(result)
            _append_jsonl(results_path, asdict(result))
    summary = summarize_results(results)
    (config.work_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def run_instance(
    instance: SweBenchInstance,
    config: SweBenchEvalConfig,
    llm_factory: LLMFactory,
    *,
    mode: str = "agent",
    isolate_mode_dir: bool = False,
    index: int = 1,
    total: int = 1,
) -> SweBenchEvalResult:
    started = time.monotonic()
    instance_dir = config.work_dir / mode / _safe_name(instance.instance_id) if isolate_mode_dir else config.work_dir / _safe_name(instance.instance_id)
    repo_dir = instance_dir / "repo"
    trace_dir = instance_dir / "trace"
    test_log_dir = instance_dir / "test_logs"
    patch_path = instance_dir / "model.patch"
    result_path = instance_dir / "result.json"
    harness: OrbitHarness | None = None
    status = "completed"
    passed: bool | None = None
    response = ""
    test_output = ""
    error: str | None = None
    stack: str | None = None

    try:
        if instance_dir.exists() and config.force:
            shutil.rmtree(instance_dir)
        if instance_dir.exists() and not config.force:
            raise FileExistsError(f"eval workspace already exists: {instance_dir}")
        instance_dir.mkdir(parents=True, exist_ok=True)
        _prepare_repo(instance, repo_dir, config)

        harness = OrbitHarness(
            HarnessConfig(
                workspace_root=repo_dir,
                trace_dir=trace_dir,
                test_log_dir=test_log_dir,
                permission_mode=config.permission_mode,
                tool_timeout_seconds=config.tool_timeout_seconds,
                max_retries=config.max_retries,
                sandbox_backend=config.sandbox_backend,
                docker_image=config.docker_image,
            ),
            approval_callback=lambda _tool, _arguments, _reason: True,
        )
        harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "instance_started", {
            "mode": mode,
            "index": index,
            "total": total,
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
        })
        if mode == "agent":
            agent = harness.create_agent(
                llm=llm_factory(instance, repo_dir, mode),
                max_context_tokens=config.max_context_tokens,
                max_rounds=config.max_rounds,
            )
            response = harness.run_chat(agent, build_agent_prompt(instance))
        elif mode == "direct":
            response = run_direct_prompt(instance, repo_dir, harness, llm_factory(instance, repo_dir, mode))
        else:
            raise ValueError(f"unsupported eval mode: {mode}")

        if config.apply_test_patch and instance.test_patch:
            _apply_test_patch(repo_dir, instance.test_patch)
            harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "test_patch_applied", {
                "instance_id": instance.instance_id,
            })

        if config.post_test_command:
            harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "verification_started", {
                "command": config.post_test_command,
            })
            test_output = harness.sandbox.run_bash(config.post_test_command, config.tool_timeout_seconds)
            passed = (
                not test_output.lstrip().startswith("Error:")
                and "[exit code:" not in test_output
            )
            if harness.sandbox.last_test_log_path is not None:
                harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "verification_log_saved", {
                    "test_log_path": str(harness.sandbox.last_test_log_path),
                })
            harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "verification_finished", {
                "passed": passed,
                "output_chars": len(test_output),
            }, status="ok" if passed else "failed")

        patch = _git(repo_dir, ["diff", "--binary"]).stdout
        patch_path.write_text(patch, encoding="utf-8")
        harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "patch_saved", {
            "patch_path": str(patch_path),
            "patch_chars": len(patch),
        })
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        stack = traceback.format_exc()
        if harness is not None:
            harness.tracer.record_error("eval", "orbit.evals.swe_bench_lite", "instance_failed", exc, {
                "instance_id": instance.instance_id,
            })
    finally:
        trace_path = None
        metrics: dict[str, Any] = {}
        test_log_path = None
        if harness is not None:
            trace_path = harness.close()
            metrics = harness.state.snapshot()
            if harness.sandbox.last_test_log_path is not None:
                test_log_path = str(harness.sandbox.last_test_log_path)
        result = SweBenchEvalResult(
            mode=mode,
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            status=status,
            passed=passed,
            duration_ms=(time.monotonic() - started) * 1000,
            workspace_path=str(repo_dir),
            patch_path=str(patch_path) if patch_path.exists() else None,
            trace_path=str(trace_path) if trace_path else None,
            result_path=str(result_path),
            response=response,
            test_command=config.post_test_command,
            test_output=test_output,
            test_log_path=test_log_path,
            error=error,
            stack=stack,
            metrics=metrics,
        )
        result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        if not config.keep_workspaces and repo_dir.exists():
            shutil.rmtree(repo_dir)
        return result


def run_direct_prompt(instance: SweBenchInstance, repo_dir: Path, harness: OrbitHarness, llm: Any) -> str:
    harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "direct_prompt_started", {
        "instance_id": instance.instance_id,
        "repo_dir": str(repo_dir),
    })
    response = llm.chat(
        [{"role": "user", "content": build_direct_prompt(instance)}],
        tools=None,
    )
    content = response.content or ""
    patch = extract_unified_diff(content)
    if not patch.strip():
        raise ValueError("direct LLM response did not contain a unified diff patch")
    patch_path = repo_dir.parent / "direct.patch"
    patch_path.write_text(patch, encoding="utf-8")
    _apply_patch(repo_dir, patch)
    harness.tracer.record("eval", "orbit.evals.swe_bench_lite", "direct_patch_applied", {
        "patch_path": str(patch_path),
        "patch_chars": len(patch),
    })
    return content


def extract_unified_diff(content: str) -> str:
    """Extract a git-apply compatible diff from an LLM response."""
    text = content.strip()
    if "```" in text:
        fenced = _extract_fenced_block(text)
        if fenced:
            text = fenced.strip()
    diff_markers = ("diff --git ", "--- ")
    for marker in diff_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[idx:]
            break
    return text.strip() + ("\n" if text.strip() else "")


def _extract_fenced_block(text: str) -> str:
    parts = text.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index]
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in {"diff", "patch"}:
            block = "\n".join(lines[1:])
        if "diff --git " in block or "--- " in block:
            return block
    return ""


def build_agent_prompt(instance: SweBenchInstance) -> str:
    fail_to_pass = json.dumps(instance.fail_to_pass, ensure_ascii=False, indent=2)
    pass_to_pass = json.dumps(instance.pass_to_pass, ensure_ascii=False, indent=2)
    return (
        "You are fixing a SWE-bench Lite issue in the checked-out repository.\n\n"
        f"Instance ID: {instance.instance_id}\n"
        f"Repository: {instance.repo}\n"
        f"Base commit: {instance.base_commit}\n\n"
        "Problem statement:\n"
        f"{instance.problem_statement}\n\n"
        "Known FAIL_TO_PASS tests from the dataset:\n"
        f"{fail_to_pass}\n\n"
        "Known PASS_TO_PASS tests from the dataset:\n"
        f"{pass_to_pass}\n\n"
        "Instructions:\n"
        "- Inspect the repository before editing.\n"
        "- Implement the smallest correct source-code change.\n"
        "- Do not commit changes.\n"
        "- Run relevant tests when possible.\n"
        "- Do not modify hidden credentials or files outside the repository.\n"
        "- Finish with a concise summary of the fix and tests run.\n"
    )


def build_direct_prompt(instance: SweBenchInstance) -> str:
    fail_to_pass = json.dumps(instance.fail_to_pass, ensure_ascii=False, indent=2)
    pass_to_pass = json.dumps(instance.pass_to_pass, ensure_ascii=False, indent=2)
    return (
        "You are given a SWE-bench Lite issue. You cannot inspect the repository with tools. "
        "Use only the prompt content below and return a single unified diff patch.\n\n"
        f"Instance ID: {instance.instance_id}\n"
        f"Repository: {instance.repo}\n"
        f"Base commit: {instance.base_commit}\n\n"
        "Problem statement:\n"
        f"{instance.problem_statement}\n\n"
        "Known FAIL_TO_PASS tests from the dataset:\n"
        f"{fail_to_pass}\n\n"
        "Known PASS_TO_PASS tests from the dataset:\n"
        f"{pass_to_pass}\n\n"
        "Output requirements:\n"
        "- Output only a unified diff patch that can be applied with `git apply`.\n"
        "- Do not include explanations, markdown outside the patch, or test logs.\n"
    )


def make_llm_factory(config: Config) -> LLMFactory:
    llm_cls = LiteLLM if config.provider == "litellm" else LLM

    def factory(_instance: SweBenchInstance, _repo_dir: Path, _mode: str):
        return llm_cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    return factory


def summarize_results(results: list[SweBenchEvalResult]) -> SweBenchEvalSummary:
    by_mode: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_mode.setdefault(result.mode, {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "unknown": 0,
            "success_rate": 0.0,
            "avg_duration_ms": 0.0,
        })
        bucket["total"] += 1
        bucket["avg_duration_ms"] += result.duration_ms
        if result.passed is True:
            bucket["passed"] += 1
        elif result.status == "failed" or result.passed is False:
            bucket["failed"] += 1
        else:
            bucket["unknown"] += 1

    for bucket in by_mode.values():
        total = bucket["total"]
        bucket["success_rate"] = bucket["passed"] / total if total else 0.0
        bucket["avg_duration_ms"] = bucket["avg_duration_ms"] / total if total else 0.0
    return SweBenchEvalSummary(total=len(results), by_mode=by_mode)


def format_summary(summary: SweBenchEvalSummary) -> str:
    lines = [f"Results: total={summary.total}"]
    for mode, bucket in sorted(summary.by_mode.items()):
        lines.append(
            f"{mode}: success_rate={bucket['success_rate']:.2%} "
            f"passed={bucket['passed']} failed={bucket['failed']} "
            f"unknown={bucket['unknown']} total={bucket['total']}"
        )
    return "\n".join(lines)


def _load_rows(source: str, split: str) -> Iterable[dict[str, Any]]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.suffix == ".jsonl":
            with source_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
            return
        if isinstance(data, dict) and isinstance(data.get("instances"), list):
            yield from data["instances"]
            return
        raise ValueError(f"unsupported local dataset shape: {source_path}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading SWE-bench Lite from Hugging Face requires the optional dependency: "
            "python -m pip install '.[eval]'"
        ) from exc
    dataset = load_dataset(source, split=split)
    for row in dataset:
        yield dict(row)


def _normalize_instance(row: dict[str, Any]) -> SweBenchInstance:
    instance_id = _required_str(row, "instance_id")
    return SweBenchInstance(
        instance_id=instance_id,
        repo=_required_str(row, "repo"),
        base_commit=_required_str(row, "base_commit"),
        problem_statement=_required_str(row, "problem_statement"),
        test_patch=str(row.get("test_patch") or ""),
        fail_to_pass=_json_list(row.get("FAIL_TO_PASS") or row.get("fail_to_pass")),
        pass_to_pass=_json_list(row.get("PASS_TO_PASS") or row.get("pass_to_pass")),
        raw=row,
    )


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset row is missing required string field: {key}")
    return value


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def _prepare_repo(instance: SweBenchInstance, repo_dir: Path, config: SweBenchEvalConfig) -> None:
    if config.repo_cache_dir is not None:
        cache_repo = Path(config.repo_cache_dir).expanduser().resolve() / _safe_name(instance.repo)
        if not cache_repo.exists():
            cache_repo.parent.mkdir(parents=True, exist_ok=True)
            _git(cache_repo.parent, ["clone", "--mirror", _repo_url(instance.repo), str(cache_repo)])
        _git(repo_dir.parent, ["clone", str(cache_repo), str(repo_dir)])
    else:
        _git(repo_dir.parent, ["clone", "--filter=blob:none", _repo_url(instance.repo), str(repo_dir)])
    _git(repo_dir, ["checkout", instance.base_commit])
    _git(repo_dir, ["status", "--short"])


def _repo_url(repo: str) -> str:
    path = Path(repo).expanduser()
    if path.exists() or repo.startswith(("http://", "https://", "git@")):
        return str(path if path.exists() else repo)
    return f"https://github.com/{repo}.git"


def _apply_test_patch(repo_dir: Path, test_patch: str) -> None:
    _apply_patch(repo_dir, test_patch)


def _apply_patch(repo_dir: Path, patch: str) -> None:
    proc = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        cwd=repo_dir,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(_format_command_failure(["git", "apply", "-"], proc))


def _git(cwd: Path, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(_format_command_failure(["git", *args], proc))
    return proc


def _format_command_failure(argv: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed ({proc.returncode}): {' '.join(argv)}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    return safe.strip(".-_") or "instance"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orbit-swe-bench-lite",
        description="Run Orbit on SWE-bench Lite instances and save patches/traces/results.",
    )
    parser.add_argument("--source", default=DEFAULT_DATASET, help="Hugging Face dataset name or local JSON/JSONL file")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, help="Maximum number of instances to run")
    parser.add_argument("--instance-id", action="append", default=[], help="Run only a specific instance ID; repeatable")
    parser.add_argument(
        "--eval-mode",
        choices=["agent", "direct", "both"],
        default="agent",
        help="agent uses Orbit tools; direct asks the LLM for a patch with prompt only; both runs both modes",
    )
    parser.add_argument("--work-dir", type=Path, default=default_eval_run_dir(), help="Directory for eval outputs")
    parser.add_argument("--repo-cache-dir", type=Path, help="Optional mirror clone cache directory")
    parser.add_argument("--no-keep-workspaces", action="store_true", help="Delete checked-out repos after each run")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-instance eval directories")
    parser.add_argument("--permission-mode", choices=[m.value for m in PermissionMode], default=PermissionMode.FULL_AUTO.value)
    parser.add_argument("--sandbox", choices=["local", "docker"], default="local")
    parser.add_argument("--docker-image", default="python:3.13-slim")
    parser.add_argument("--tool-timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-context-tokens", type=int, default=128_000)
    parser.add_argument("--apply-test-patch", action="store_true", help="Apply dataset test_patch before post-test-command")
    parser.add_argument("--post-test-command", default="", help="Optional command to run after Orbit finishes")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        llm_config = parse_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    if not llm_config.api_key:
        print("No API key found. Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or ORBIT_API_KEY.")
        return 1

    eval_config = SweBenchEvalConfig(
        source=args.source,
        split=args.split,
        limit=args.limit,
        instance_ids=args.instance_id,
        eval_mode=args.eval_mode,
        work_dir=args.work_dir,
        repo_cache_dir=args.repo_cache_dir,
        keep_workspaces=not args.no_keep_workspaces,
        force=args.force,
        permission_mode=PermissionMode(args.permission_mode),
        sandbox_backend=args.sandbox,
        docker_image=args.docker_image,
        tool_timeout_seconds=args.tool_timeout,
        max_retries=args.max_retries,
        max_rounds=args.max_rounds,
        max_context_tokens=args.max_context_tokens,
        apply_test_patch=args.apply_test_patch,
        post_test_command=args.post_test_command,
    )
    results = run_eval(eval_config, make_llm_factory(llm_config))
    summary = summarize_results(results)
    print(format_summary(summary))
    print(f"Output directory: {eval_config.work_dir}")
    any_failed = any(result.status == "failed" or result.passed is False for result in results)
    return 0 if not any_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
