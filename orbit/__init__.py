"""Orbit - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from orbit.agent import Agent
from orbit.config import Config, ConfigError, parse_config
from orbit.harness import OrbitHarness, HarnessConfig, PermissionMode, TraceRecorder
from orbit.llm import LLM
from orbit.tools import ALL_TOOLS, get_default_tools

__all__ = [
    "ALL_TOOLS",
    "get_default_tools",
    "LLM",
    "Agent",
    "Config",
    "ConfigError",
    "parse_config",
    "OrbitHarness",
    "HarnessConfig",
    "PermissionMode",
    "TraceRecorder",
    "__version__",
]
