"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """Load .env from cwd, parents, and the orbit package dir."""
    if os.getenv("ORBIT_DISABLE_DOTENV"):
        return
    try:
        from dotenv import load_dotenv

        candidates = [Path.cwd() / ".env"]
        cur = Path.cwd()
        home = Path.home()
        while cur != home and cur != cur.parent:
            candidates.append(cur / ".env")
            cur = cur.parent
        candidates.append(Path(__file__).resolve().parent / ".env")

        seen: set[Path] = set()
        for candidate in candidates:
            path = candidate.expanduser().resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            load_dotenv(path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, silently skip


def _env(name: str, default: str = "") -> str:
    legacy_name = f"ORBIT_{name.removeprefix('ORBIT_')}"
    return os.getenv(name) or os.getenv(legacy_name) or default


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"
    permission_mode: str = "default"
    workspace_root: str = ""
    trace_dir: str = ""
    test_log_dir: str = ""
    tool_timeout_seconds: int = 60
    max_retries: int = 0
    sandbox_backend: str = "local"
    docker_image: str = "python:3.13-slim"
    # MCP配置文件路径；为空时走ORBIT_MCP_CONFIG或工作区默认配置文件发现。
    mcp_config_file: str = ""
    # MCP总开关；关闭后Agent只加载内置工具，不启动任何MCP服务。
    mcp_enabled: bool = True
    # skills目录；为空时使用默认目录发现规则。
    skills_dir: str = ""
    # skills总开关；关闭后不会注入技能目录，也不会暴露load_skill工具。
    skills_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            _env("ORBIT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=_env("ORBIT_MODEL", "gpt-5.5"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or _env("ORBIT_BASE_URL"),
            max_tokens=int(_env("ORBIT_MAX_TOKENS", "8192")),
            temperature=float(_env("ORBIT_TEMPERATURE", "0")),
            max_context_tokens=int(_env("ORBIT_MAX_CONTEXT", "128000")),
            provider=_env("ORBIT_PROVIDER", "openai"),
            permission_mode=_env("ORBIT_PERMISSION_MODE", "default"),
            workspace_root=_env("ORBIT_WORKSPACE_ROOT", ""),
            trace_dir=_env("ORBIT_TRACE_DIR", ""),
            test_log_dir=_env("ORBIT_TEST_LOG_DIR", ""),
            tool_timeout_seconds=int(_env("ORBIT_TOOL_TIMEOUT", "60")),
            max_retries=int(_env("ORBIT_MAX_RETRIES", "0")),
            sandbox_backend=_env("ORBIT_SANDBOX", "local"),
            docker_image=_env("ORBIT_DOCKER_IMAGE", "python:3.13-slim"),
            # MCP配置从环境变量进入Config，CLI参数会在main里覆盖这里的值。
            mcp_config_file=_env("ORBIT_MCP_CONFIG_FILE", ""),
            mcp_enabled=_env("ORBIT_MCP_DISABLED", "").lower() not in {"1", "true", "yes"},
            skills_dir=_env("ORBIT_SKILLS_DIR", ""),
            skills_enabled=_env("ORBIT_SKILLS_DISABLED", "").lower() not in {"1", "true", "yes"},
        )


class ConfigError(Exception):
    """Raised when Orbit configuration cannot be parsed."""


def parse_config() -> Config:
    try:
        return Config.from_env()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
