"""Sandbox boundary and command runners for Orbit harness."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class SandboxError(RuntimeError):
    """Raised when sandbox execution cannot be completed."""


def validate_workspace_path(path: str | Path, workspace_root: str | Path) -> tuple[bool, str, Path]:
    resolved = Path(path).expanduser().resolve()
    root = Path(workspace_root).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return False, f"path {resolved} is outside workspace boundary ({root})", resolved
    return True, "", resolved


@dataclass
class SandboxConfig:
    backend: str = "local"
    docker_image: str = "python:3.13-slim"
    network_enabled: bool = False
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    extra_mounts: list[str] = field(default_factory=list)
    test_log_dir: Path | None = None


class SandboxRunner:
    """Executes shell commands either locally or in a Docker sandbox."""

    def __init__(self, config: SandboxConfig, workspace_root: str | Path):
        self.config = config
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.last_test_log_path: Path | None = None

    def run_bash(self, command: str, timeout: int) -> str:
        if self.config.backend == "docker":
            return self._run_bash_in_docker(command, timeout)
        if self.config.backend == "local":
            return self._run_bash_locally(command, timeout)
        raise SandboxError(f"unsupported sandbox backend: {self.config.backend}")

    def _run_bash_locally(self, command: str, timeout: int) -> str:
        proc = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=self.workspace_root,
        )
        output = _format_process_output(proc.stdout, proc.stderr, proc.returncode)
        self._write_test_log_if_needed(command, proc.stdout, proc.stderr, proc.returncode, output)
        return output

    def _run_bash_in_docker(self, command: str, timeout: int) -> str:
        network = "bridge" if self.config.network_enabled else "none"
        root = str(self.workspace_root)
        argv = [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "--cpus",
            str(self.config.cpu_limit),
            "--memory",
            self.config.memory_limit,
            "-v",
            f"{root}:{root}:rw",
            "-w",
            root,
        ]
        for mount in self.config.extra_mounts:
            argv.extend(["-v", mount])
        argv.extend([self.config.docker_image, "bash", "-lc", command])

        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = _format_process_output(proc.stdout, proc.stderr, proc.returncode)
        self._write_test_log_if_needed(command, proc.stdout, proc.stderr, proc.returncode, output)
        return output

    def _write_test_log_if_needed(
        self,
        command: str,
        stdout: str,
        stderr: str,
        returncode: int,
        formatted_output: str,
    ) -> None:
        if not _looks_like_test_command(command):
            return
        log_dir = self.config.test_log_dir or default_test_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = log_dir / f"test-run-{timestamp}-{uuid4().hex[:8]}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(self.workspace_root),
            "sandbox_backend": self.config.backend,
            "command": command,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "formatted_output": formatted_output,
        }
        path.write_text(_json_dumps(payload), encoding="utf-8")
        self.last_test_log_path = path


def _format_process_output(stdout: str, stderr: str, returncode: int) -> str:
    out = stdout
    if stderr:
        out += f"\n[stderr]\n{stderr}"
    if returncode != 0:
        out += f"\n[exit code: {returncode}]"
    if len(out) > 15_000:
        out = out[:6000] + f"\n\n... truncated ({len(out)} chars total) ...\n\n" + out[-3000:]
    return out.strip() or "(no output)"


def default_test_log_dir() -> Path:
    date_dir = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(__file__).resolve().parents[2] / "tests" / "logs" / date_dir


def _looks_like_test_command(command: str) -> bool:
    lowered = command.lower()
    markers = (
        "pytest",
        "python -m pytest",
        "unittest",
        "python -m unittest",
        "coverage run",
        "tox ",
        "nox ",
    )
    return any(marker in lowered for marker in markers)


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))
