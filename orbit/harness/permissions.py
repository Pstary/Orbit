"""Permission and access-control policy for Orbit harness."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    FULL_AUTO = "full_auto"


SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    "*/.ssh/*",
    "*/.aws/credentials",
    "*/.aws/config",
    "*/.config/gcloud/*",
    "*/.azure/*",
    "*/.gnupg/*",
    "*/.docker/config.json",
    "*/.kube/config",
    "*/.env",
    "*/.orbit/credentials.json",
)

DEFAULT_PROTECTED_PATTERNS: tuple[str, ...] = (
    "orbit/agent.py",
    "orbit/harness/*",
    "orbit/tools/*",
    "orbit/config.py",
    "pyproject.toml",
)

DEFAULT_DENIED_COMMAND_PATTERNS: tuple[str, ...] = (
    r"\brm\b(?=.*\s-[^\s]*[rR])(?=.*\s-[^\s]*f)",
    r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r">\s*/dev/sd[a-z]",
    r"\bchmod\s+(-R\s+)?777\s+/",
    r":\(\)\s*\{.*:\|:.*\}",
    r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b",
    r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh\b",
)

READ_ONLY_TOOLS = {"read_file", "grep", "glob"}
MUTATING_TOOLS = {"bash", "write_file", "edit_file"}


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str = ""


@dataclass
class PermissionSettings:
    mode: PermissionMode = PermissionMode.DEFAULT
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    protected_patterns: tuple[str, ...] = DEFAULT_PROTECTED_PATTERNS
    denied_command_patterns: tuple[str, ...] = DEFAULT_DENIED_COMMAND_PATTERNS
    sensitive_path_patterns: tuple[str, ...] = SENSITIVE_PATH_PATTERNS


class PolicyEngine:
    """Central permission checker used before every tool execution."""

    def __init__(self, settings: PermissionSettings, workspace_root: str | Path):
        self.settings = settings
        self.workspace_root = Path(workspace_root).expanduser().resolve()

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        file_path: Path | None = None,
        command: str | None = None,
    ) -> PermissionDecision:
        del arguments
        if tool_name in self.settings.denied_tools:
            return PermissionDecision(False, reason=f"{tool_name} is explicitly denied")

        if file_path is not None:
            sensitive = self._sensitive_path_reason(file_path)
            if sensitive:
                return PermissionDecision(False, reason=sensitive)

            if self._is_protected_path(file_path):
                return PermissionDecision(
                    allowed=False,
                    requires_approval=True,
                    reason=f"{file_path} matches protected path policy",
                )

        if command:
            denied = self._denied_command_reason(command)
            if denied:
                return PermissionDecision(False, reason=denied)

        if tool_name in self.settings.allowed_tools:
            return PermissionDecision(True, reason=f"{tool_name} is explicitly allowed")

        is_read_only = tool_name in READ_ONLY_TOOLS
        if self.settings.mode == PermissionMode.FULL_AUTO:
            return PermissionDecision(True, reason="full_auto mode allows this tool")

        if is_read_only:
            return PermissionDecision(True, reason="read-only tool is allowed")

        if self.settings.mode == PermissionMode.PLAN:
            return PermissionDecision(False, reason="plan mode blocks mutating tools")

        if tool_name in MUTATING_TOOLS:
            return PermissionDecision(
                allowed=False,
                requires_approval=True,
                reason=f"{tool_name} may mutate the workspace and requires approval",
            )

        return PermissionDecision(
            allowed=False,
            requires_approval=True,
            reason=f"{tool_name} is not known to be read-only and requires approval",
        )

    def _sensitive_path_reason(self, path: Path) -> str:
        path_texts = _policy_match_paths(str(path))
        for path_text in path_texts:
            for pattern in self.settings.sensitive_path_patterns:
                if fnmatch.fnmatch(path_text, pattern):
                    return f"Access denied: {path} matches sensitive path pattern {pattern}"
        return ""

    def _is_protected_path(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.workspace_root)
        except ValueError:
            return False
        rel_text = rel.as_posix()
        return any(fnmatch.fnmatch(rel_text, pattern) for pattern in self.settings.protected_patterns)

    def _denied_command_reason(self, command: str) -> str:
        for pattern in self.settings.denied_command_patterns:
            if re.search(pattern, command):
                return f"Command denied by policy pattern: {pattern}"
        return ""


def _policy_match_paths(file_path: str) -> tuple[str, ...]:
    normalized = file_path.rstrip("/")
    if not normalized:
        return (file_path,)
    return (normalized, normalized + "/")
