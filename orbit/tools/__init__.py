"""工具注册表。"""

import os

from .agent import AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .fetch import FetchUrlTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .read import ReadFileTool
from .write import WriteFileTool

ALL_TOOLS = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    FetchUrlTool(),
]


def get_default_tools(*, include_mcp: bool = True, mcp_config_path: str | None = None):
    """返回内置工具和配置发现到的MCP工具。

    MCP发现必须显式发生在这个函数边界，避免单纯import orbit.tools时启动外部进程。
    """

    tools = list(ALL_TOOLS)
    if include_mcp and os.getenv("ORBIT_MCP_DISABLED", "").lower() not in {"1", "true", "yes"}:
        # 延迟导入MCP模块，保证没有MCP配置的普通路径仍然轻量。
        from ..mcp import discover_mcp_tools

        tools.extend(discover_mcp_tools(mcp_config_path))
    return tools


def get_tool(name: str):
    """按工具名查找内置工具。"""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
