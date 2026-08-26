"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from corecoder.agent import Agent
from corecoder.config import Config, ConfigError, parse_config
from corecoder.llm import LLM
from corecoder.tools import ALL_TOOLS

__all__ = ["ALL_TOOLS", "LLM", "Agent", "Config", "ConfigError", "parse_config", "__version__"]
