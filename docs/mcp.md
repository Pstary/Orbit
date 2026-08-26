# MCP接入说明

Orbit可以加载Model Context Protocol stdio服务，并把远端工具暴露成普通Orbit工具。Agent循环不直接处理MCP协议：MCP工具会先适配为现有`Tool`接口，再进入Harness的权限、超时、重试和trace链路。

## 配置发现

Orbit按以下顺序发现MCP服务：

1. `ORBIT_MCP_CONFIG`，内容是JSON对象。
2. `ORBIT_MCP_CONFIG_FILE`或CLI参数`--mcp-config`指定的路径。
3. 当前工作区下的`.mcp.json`、`mcp.json`或`.cursor/mcp.json`。

关闭MCP自动发现：

```bash
export ORBIT_MCP_DISABLED=1
orbit --no-mcp
```

## 配置格式

支持Cursor风格的`mcpServers`配置：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "timeout_seconds": 10
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "..."
      }
    }
  }
}
```

也支持列表形式，下面仍然以真实GitHubMCP服务为例：

```json
{
  "servers": [
    {
      "name": "github",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "timeout_seconds": 20
    }
  ]
}
```

## 工具命名

MCP工具会按以下格式暴露给模型：

```text
mcp__<server_name>__<tool_name>
```

不符合`[a-zA-Z0-9_-]`的字符会转换成`_`，过长名称会用稳定hash后缀截断。例如，GitHub服务下的`search_repositories`会变成：

```text
mcp__github__search_repositories
```

## 调用MCP工具

配置完成后正常启动Orbit：

```bash
orbit --mcp-config .mcp.json
```

MCP工具会和内置工具一起进入同一份function-calling schema，模型可以自然选择调用。程序化用法：

```python
from orbit import Agent, LLM, get_default_tools

llm = LLM(api_key="...", model="gpt-5.5")
tools = get_default_tools(include_mcp=True, mcp_config_path=".mcp.json")
agent = Agent(llm=llm, tools=tools)

agent.chat("使用可用的MCP工具检查外部服务。")
```

## 错误处理与治理

- 发现失败不会导致Agent启动失败。Orbit会暴露一个状态工具，例如`mcp__github__status`，调用后返回连接错误。
- MCP调用会经过Harness的超时和重试处理。
- 默认权限模式下，MCP工具会被视为未知外部工具，需要人工审批。只有在信任服务时才建议使用`ORBIT_PERMISSION_MODE=full_auto`。
- 所有调用都会记录到Harness trace中，包含暴露给模型的工具名和脱敏后的参数。

## 真实服务连通性验证

可以直接用真实GitHubMCP服务打印schema并调用只读工具：

```bash
python - <<'PY'
import json
import tempfile

from orbit.harness import HarnessConfig, OrbitHarness, PermissionMode
from orbit.llm import ToolCall
from orbit.mcp import discover_mcp_tools

cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "timeout_seconds": 20
        }
    }
}, cfg)
cfg.close()

tools = {tool.name: tool for tool in discover_mcp_tools(cfg.name)}
tool = tools["mcp__github__search_repositories"]

print(json.dumps(tool.schema(), ensure_ascii=False, indent=2))

harness = OrbitHarness(HarnessConfig(
    permission_mode=PermissionMode.FULL_AUTO,
    tool_timeout_seconds=30,
))
print(harness.execute_tool_call(
    tool,
    ToolCall(
        id="github-search-1",
        name=tool.name,
        arguments={"query": "modelcontextprotocol/mcp", "page": 1, "perPage": 3},
    ),
))
PY
```

这条链路会真实启动GitHubMCP服务，发现`mcp__github__search_repositories`等工具，并通过Harness执行一次GitHub仓库搜索。
