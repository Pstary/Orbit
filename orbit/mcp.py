"""MCP客户端接入层。

这里负责把MCPstdio服务适配成Orbit现有Tool接口。Agent层仍然只认识普通工具；
服务发现、协议收发和工具命名转换都收敛在这个模块里。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools.base import Tool

MCP_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_CONFIG_FILES = (".mcp.json", "mcp.json", ".cursor/mcp.json")
TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class McpError(Exception):
    """MCP服务发现或调用失败时抛出的边界异常。"""


@dataclass
class McpServerConfig:
    # 单个MCP服务的启动配置，字段基本对应.mcp.json里的server声明。
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_seconds: float = 10.0
    disabled: bool = False
    framing: str = "auto"


@dataclass
class McpToolSpec:
    # 从MCP服务tools/list响应里解析出的工具元信息。
    server_name: str
    tool_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]


def load_mcp_server_configs(config_path: str | Path | None = None) -> list[McpServerConfig]:
    """从环境变量或工作区配置文件加载MCP服务定义。

    支持两种配置形态：
    - {"mcpServers": {"name": {"command": "...", "args": [...]}}}
    - {"servers": [{"name": "name", "command": "...", "args": [...]}]}
    """

    payload = _load_config_payload(config_path)
    if not payload:
        return []

    # 兼容Cursor常用的mcpServers对象结构，也兼容显式servers列表。
    raw_servers: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload.get("mcpServers"), dict):
        raw_servers.extend((name, cfg) for name, cfg in payload["mcpServers"].items() if isinstance(cfg, dict))
    if isinstance(payload.get("servers"), list):
        for index, cfg in enumerate(payload["servers"]):
            if isinstance(cfg, dict):
                raw_servers.append((str(cfg.get("name") or f"server_{index + 1}"), cfg))

    servers: list[McpServerConfig] = []
    for name, cfg in raw_servers:
        # command为空的服务没有启动入口，直接跳过，避免后面Popen报低质量错误。
        command = str(cfg.get("command") or "").strip()
        if not command:
            continue
        args = cfg.get("args") or []
        if not isinstance(args, list):
            raise McpError(f"MCP server {name} has invalid args; expected a list")
        env = cfg.get("env") or {}
        if not isinstance(env, dict):
            raise McpError(f"MCP server {name} has invalid env; expected an object")
        timeout = float(cfg.get("timeout_seconds") or cfg.get("timeout") or 10)
        servers.append(McpServerConfig(
            name=_safe_segment(name),
            command=command,
            args=[str(item) for item in args],
            env={str(k): str(v) for k, v in env.items()},
            cwd=str(cfg["cwd"]) if cfg.get("cwd") else None,
            timeout_seconds=timeout,
            disabled=bool(cfg.get("disabled", False)),
            framing=str(cfg.get("framing") or "auto"),
        ))
    return servers


def discover_mcp_tools(config_path: str | Path | None = None) -> list[Tool]:
    """发现已配置MCP服务，并返回可直接交给Agent使用的Orbit工具适配器。"""

    tools: list[Tool] = []
    seen_names: set[str] = set()
    for server in load_mcp_server_configs(config_path):
        if server.disabled:
            continue
        try:
            client = McpStdioClient(server)
            specs = client.list_tools()
            # 不同服务或不同远端工具可能在清洗后重名，这里统一做稳定去重。
            for spec in specs:
                spec.exposed_name = _unique_name(spec.exposed_name, seen_names)
            tools.extend(McpTool(client, spec) for spec in specs)
        except Exception as exc:  # noqa: BLE001
            # 发现失败时不让Agent启动失败，而是暴露一个status工具让用户看到原因。
            status = McpStatusTool(server.name, f"Error discovering MCP server {server.name}: {exc}")
            status.name = _unique_name(status.name, seen_names)
            tools.append(status)
    return tools


class McpStdioClient:
    """轻量MCPstdioJSON-RPC客户端。

    每个MCP服务保持一个子进程，并用锁串行化请求。这样能贴合当前Harness模型：
    一次工具调用有超时边界，并最终返回一段文本观察结果。
    """

    def __init__(self, config: McpServerConfig):
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._framing = config.framing if config.framing in {"headers", "jsonl"} else "headers"

    def list_tools(self) -> list[McpToolSpec]:
        # tools/list是MCP工具发现入口，返回值会被转换成OpenAIfunction-calling schema。
        self._ensure_started()
        result = self._request("tools/list", {})
        tools = result.get("tools") or []
        specs: list[McpToolSpec] = []
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            tool_name = str(raw.get("name") or "").strip()
            if not tool_name:
                continue
            exposed = _exposed_tool_name(self.config.name, tool_name)
            input_schema = raw.get("inputSchema") or {"type": "object", "properties": {}, "required": []}
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}, "required": []}
            # Orbit的Tool.schema要求parameters至少是object/properties/required三件套。
            input_schema.setdefault("type", "object")
            input_schema.setdefault("properties", {})
            input_schema.setdefault("required", [])
            description = str(raw.get("description") or f"MCP tool {tool_name} from server {self.config.name}")
            specs.append(McpToolSpec(
                server_name=self.config.name,
                tool_name=tool_name,
                exposed_name=exposed,
                description=description,
                input_schema=input_schema,
            ))
        return specs

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        # tools/call返回MCPcontent数组；这里统一压成Agent能读的字符串。
        self._ensure_started()
        result = self._request("tools/call", {"name": tool_name, "arguments": arguments})
        if result.get("isError"):
            return "MCP tool error: " + _format_mcp_content(result)
        return _format_mcp_content(result)

    def close(self) -> None:
        # MCP服务是长驻子进程；工具对象释放或显式关闭时要尽量回收进程。
        proc = self._process
        self._process = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __del__(self) -> None:
        self.close()

    def _ensure_started(self) -> None:
        if self._process and self._process.poll() is None:
            return
        # 标准MCPstdio使用Content-Length帧；部分历史服务使用JSON-lines。
        # auto模式先走标准帧，初始化失败后再降级到JSON-lines。
        if self.config.framing == "jsonl":
            self._framing = "jsonl"
        else:
            self._framing = "headers"
        try:
            self._start_process()
            self._initialize()
        except Exception:
            if self.config.framing != "auto" or self._framing == "jsonl":
                raise
            self.close()
            self._framing = "jsonl"
            self._start_process()
            self._initialize()

    def _start_process(self) -> None:
        env = os.environ.copy()
        env.update(self.config.env)
        # stdin/stdout必须保持二进制模式，因为Content-Length按UTF-8字节数计算。
        self._process = subprocess.Popen(
            [self.config.command, *self.config.args],
            cwd=self.config.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _initialize(self) -> None:
        # MCP连接建立后先initialize，再发送initialized通知，之后才能list/call工具。
        self._request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "orbit", "version": "0.4.0"},
        })
        self._notify("notifications/initialized", {})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # MCP是JSON-RPC协议，请求id用于把响应和当前调用配对。
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            deadline = time.monotonic() + self.config.timeout_seconds
            while True:
                if self._process and self._process.poll() is not None:
                    raise McpError(f"server exited with code {self._process.returncode}; stderr={self._stderr_text()}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"MCP request {method} timed out after {self.config.timeout_seconds}s")
                message = self._read_message(deadline)
                # 服务端可能先发notification；没有匹配当前id的消息直接跳过。
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    if isinstance(error, dict):
                        raise McpError(str(error.get("message") or error))
                    raise McpError(str(error))
                result = message.get("result") or {}
                if not isinstance(result, dict):
                    raise McpError(f"MCP response for {method} is not an object")
                return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        # notification没有id，也不会等待响应。
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise McpError("MCP server process is not running")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        body_bytes = body.encode("utf-8")
        if self._framing == "jsonl":
            # 兼容旧版GitHubMCP这类JSON-lines服务。
            self._process.stdin.write(body_bytes + b"\n")
        else:
            # 标准MCPstdio帧格式：HTTP风格header加JSONbody。
            self._process.stdin.write(f"Content-Length: {len(body_bytes)}\r\n\r\n".encode("ascii") + body_bytes)
        self._process.stdin.flush()

    def _read_message(self, deadline: float) -> dict[str, Any]:
        if not self._process or not self._process.stdout:
            raise McpError("MCP server process is not running")

        if self._framing == "jsonl":
            # JSON-lines模式下一行就是一个完整JSON-RPC消息。
            line = self._readline(deadline).strip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise McpError(f"MCP response is not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise McpError("MCP response is not a JSON object")
            return data

        headers: dict[str, str] = {}
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("MCP response timed out while reading headers")
            line = self._readline(deadline)
            if line == "":
                raise McpError(f"MCP server closed stdout; stderr={self._stderr_text()}")
            line = line.rstrip("\r\n")
            if not line:
                break
            if ":" not in line:
                # 少量历史服务会直接吐JSON行；读到这种格式时尽量兼容。
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()

        length_text = headers.get("content-length")
        if not length_text:
            raise McpError("MCP response missing Content-Length")
        body = self._read_exact(int(length_text), deadline).decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise McpError(f"MCP response is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise McpError("MCP response is not a JSON object")
        return data

    def _readline(self, deadline: float) -> str:
        if not self._process or not self._process.stdout:
            raise McpError("MCP server process is not running")
        if os.name != "nt":
            # macOS/Linux下用select给阻塞读加deadline，避免坏服务卡死启动。
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError("MCP response timed out while reading line")
        return self._process.stdout.readline().decode("utf-8")

    def _read_exact(self, length: int, deadline: float) -> bytes:
        if not self._process or not self._process.stdout:
            raise McpError("MCP server process is not running")
        chunks: list[bytes] = []
        remaining_length = length
        while remaining_length > 0:
            if os.name != "nt":
                # Content-Length按字节读，不能用文本read，否则中文等多字节内容会错位。
                remaining_time = max(0.0, deadline - time.monotonic())
                readable, _, _ = select.select([self._process.stdout], [], [], remaining_time)
                if not readable:
                    raise TimeoutError("MCP response timed out while reading body")
            chunk = self._process.stdout.read(remaining_length)
            if chunk == b"":
                raise McpError(f"MCP server closed stdout; stderr={self._stderr_text()}")
            chunks.append(chunk)
            remaining_length -= len(chunk)
        return b"".join(chunks)

    def _drain_stderr(self) -> None:
        # stderr只保留尾部，出错时给用户足够线索，同时避免trace膨胀。
        proc = self._process
        if not proc or not proc.stderr:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip("\n"))
            if len(self._stderr_tail) > 20:
                self._stderr_tail = self._stderr_tail[-20:]

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail[-5:])


class McpTool(Tool):
    """把单个MCP远端工具包装成OrbitTool。"""

    def __init__(self, client: McpStdioClient, spec: McpToolSpec):
        self.client = client
        self.server_name = spec.server_name
        self.remote_tool_name = spec.tool_name
        self.name = spec.exposed_name
        self.description = f"[MCP:{spec.server_name}] {spec.description}"
        self.parameters = spec.input_schema

    def execute(self, **kwargs) -> str:
        try:
            return self.client.call_tool(self.remote_tool_name, kwargs)
        except Exception as exc:  # noqa: BLE001
            return f"Error executing MCP tool {self.name}: {exc}"


class McpStatusTool(Tool):
    """MCP服务发现失败时暴露给Agent的可见状态工具。"""

    def __init__(self, server_name: str, message: str):
        self.name = _exposed_tool_name(server_name, "status")
        self.description = f"MCP server {server_name} failed to load. Call this tool to see the error."
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self._message = message

    def execute(self) -> str:
        return self._message


def _load_config_payload(config_path: str | Path | None) -> dict[str, Any]:
    # ORBIT_MCP_CONFIG优先级最高，适合在CI或临时命令里注入完整JSON。
    raw = os.getenv("ORBIT_MCP_CONFIG")
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise McpError(f"ORBIT_MCP_CONFIG is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise McpError("ORBIT_MCP_CONFIG must be a JSON object")
        return payload

    # 显式文件路径优先于默认工作区文件；路径不存在时直接报配置错误。
    configured_path = config_path or os.getenv("ORBIT_MCP_CONFIG_FILE")
    if configured_path:
        path = Path(configured_path)
        path = path.expanduser()
        if path.exists():
            return _read_json_object(path)
        raise McpError(f"MCP config file not found: {path}")

    # 最后才扫描工作区常见MCP配置文件。
    for candidate in DEFAULT_CONFIG_FILES:
        found = Path.cwd() / candidate
        if found.exists():
            return _read_json_object(found)
    return {}


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpError(f"MCP config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise McpError(f"MCP config file {path} must contain a JSON object")
    return payload


def _format_mcp_content(result: dict[str, Any]) -> str:
    # MCPcontent可以包含text/image/resource等类型；当前Agent只吃文本，所以非文本转JSON。
    content = result.get("content")
    if not isinstance(content, list):
        return json.dumps(result, ensure_ascii=False, indent=2)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(json.dumps(item, ensure_ascii=False, indent=2))
    return "\n".join(part for part in parts if part)


def _safe_segment(value: str) -> str:
    # OpenAI工具名只允许有限字符集，服务名和工具名都要清洗。
    segment = TOOL_NAME_RE.sub("_", value.strip()).strip("_")
    return segment or "mcp"


def _exposed_tool_name(server_name: str, tool_name: str) -> str:
    # 用server+tool组成全局工具名，避免多个MCP服务的远端工具互相覆盖。
    name = f"mcp__{_safe_segment(server_name)}__{_safe_segment(tool_name)}"
    if len(name) <= 64:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[:55]}_{digest}"


def _unique_name(name: str, seen: set[str]) -> str:
    # 清洗或截断后仍可能撞名，用hash后缀做稳定去重。
    if name not in seen:
        seen.add(name)
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    candidate = f"{name[:55]}_{digest}"
    suffix = 2
    while candidate in seen:
        candidate = f"{name[:52]}_{digest}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate
