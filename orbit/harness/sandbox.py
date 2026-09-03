"""Sandbox boundary and command runners for Orbit harness."""

from __future__ import annotations

import shlex
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# getuid/getgid are Unix-only; on Windows the docker backend runs as the
# image's default user instead of mapping the host uid/gid.
if sys.platform != "win32":
    from os import getgid, getuid
else:
    getuid = None
    getgid = None


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
    pids_limit: int = 256
    read_only_rootfs: bool = True
    seccomp_profile: str = ""
    extra_mounts: list[str] = field(default_factory=list)
    test_log_dir: Path | None = None


class SandboxRunner:
    """Executes shell commands either locally or in a Docker sandbox."""

    def __init__(self, config: SandboxConfig, workspace_root: str | Path):
        self.config = config
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.last_test_log_path: Path | None = None

    def run_bash(self, command: str, timeout: int, cancellation=None) -> str:
        if self.config.backend == "docker":
            return self._run_bash_in_docker(command, timeout, cancellation)
        if self.config.backend == "local":
            return self._run_bash_locally(command, timeout, cancellation)
        raise SandboxError(f"unsupported sandbox backend: {self.config.backend}")

    def _run_bash_locally(self, command: str, timeout: int, cancellation=None) -> str:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.workspace_root,
            **_process_group_options(),
        )
        stdout, stderr = _communicate_interruptibly(proc, timeout, cancellation)
        output = _format_process_output(stdout, stderr, proc.returncode)
        self._write_test_log_if_needed(command, stdout, stderr, proc.returncode, output)
        return output

    def _run_bash_in_docker(self, command: str, timeout: int, cancellation=None) -> str:
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
            "--pids-limit",
            str(self.config.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if getuid is not None:
            argv.extend(["--user", f"{getuid()}:{getgid()}"])
        argv += [
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--tmpfs",
            "/run:rw,nosuid,nodev,noexec,size=16m",
            "-v",
            f"{root}:{root}:rw",
            "-w",
            root,
        ]
        if self.config.read_only_rootfs:
            argv.append("--read-only")
        if self.config.seccomp_profile:
            argv.extend(["--security-opt", f"seccomp={self.config.seccomp_profile}"])
        for mount in self.config.extra_mounts:
            argv.extend(["-v", mount])
        argv.extend([self.config.docker_image, "bash", "-lc", command])

        try:
            # Preserve the simple direct-run API for callers that do not opt in
            # to cancellation (and for compatibility with custom integrations).
            if cancellation is None:
                completed = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
                output = _format_process_output(
                    completed.stdout, completed.stderr, completed.returncode,
                )
                self._write_test_log_if_needed(
                    command, completed.stdout, completed.stderr, completed.returncode, output,
                )
                return output
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_process_group_options(),
            )
            stdout, stderr = _communicate_interruptibly(proc, timeout, cancellation)
        except FileNotFoundError:
            return "Error: docker executable not found. Install Docker and ensure `docker` is on PATH."
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"
        output = _format_process_output(stdout, stderr, proc.returncode)
        self._write_test_log_if_needed(command, stdout, stderr, proc.returncode, output)
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


def _process_group_options() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Stop the command and descendants, then reap the direct child."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
        else:
            proc.kill()
        proc.wait()


def _communicate_interruptibly(proc: subprocess.Popen, timeout: int, cancellation=None) -> tuple[str, str]:
    from ..cancellation import ToolInterrupted

    deadline = time.monotonic() + timeout
    while True:
        if cancellation is not None and cancellation.cancelled:
            _terminate_process_tree(proc)
            proc.communicate()
            raise ToolInterrupted("command interrupted by user")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_tree(proc)
            proc.communicate()
            raise TimeoutError(f"command timed out after {timeout}s")
        try:
            return proc.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue


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
