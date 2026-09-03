"""Interactive REPL - the user-facing terminal interface."""

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import Agent
from .cancellation import ToolInterrupted
from .config import Config, ConfigError, parse_config
from .harness import OrbitHarness, HarnessConfig, PermissionMode
from .llm import LLM, LiteLLM
from .skill_registry import default_skills_dir

console = Console()


def _run_chat_interruptibly(agent: Agent, user_input: str, **callbacks) -> str:
    """Run a turn off the UI thread so Ctrl+C can request cooperative cancellation."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="orbit-agent")
    future = pool.submit(agent.harness.run_chat, agent, user_input, **callbacks)
    interrupt_requested = False
    try:
        while True:
            try:
                return future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                continue
            except KeyboardInterrupt:
                if interrupt_requested:
                    raise
                interrupt_requested = True
                agent.harness.interrupt(agent)
                console.print("\n[yellow]Stopping the active operation...[/yellow]")
    finally:
        # Do not return to the prompt while an old tool can still mutate files.
        pool.shutdown(wait=True, cancel_futures=True)


def _parse_args():
    p = argparse.ArgumentParser(
        prog="orbit",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $ORBIT_MODEL or gpt-5.5)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("--max-tokens", type=int, help="Maximum output tokens per model call")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("--demo", action="store_true", help="Run the offline scripted demo (no API key needed)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("--permission-mode", choices=[m.value for m in PermissionMode], help="Harness permission mode")
    p.add_argument("--workspace-root", help="Workspace root for harness path isolation")
    p.add_argument("--trace-dir", help="Directory where harness trace files are written")
    p.add_argument("--test-log-dir", help="Directory where harness test command logs are written")
    p.add_argument("--tool-timeout", type=int, help="Maximum seconds for one tool execution")
    p.add_argument("--max-retries", type=int, help="Maximum retries for failed tool execution")
    p.add_argument("--sandbox", choices=["local", "docker"], help="Harness sandbox backend")
    p.add_argument("--docker-image", help="Docker image used by the docker sandbox backend")
    p.add_argument("--docker-network", action="store_true", help="Enable network access inside Docker sandbox")
    p.add_argument("--docker-cpus", type=float, help="CPU limit for Docker sandbox")
    p.add_argument("--docker-memory", help="Memory limit for Docker sandbox, e.g. 512m or 2g")
    p.add_argument("--docker-pids-limit", type=int, help="Process limit for Docker sandbox")
    p.add_argument("--docker-writable-rootfs", action="store_true", help="Disable Docker read-only root filesystem")
    p.add_argument("--docker-seccomp-profile", help="Path to a custom Docker seccomp profile")
    # MCP相关参数只影响工具发现，不改变Agent主循环和Harness执行模型。
    p.add_argument("--mcp-config", help="Path to MCP server config (default: $ORBIT_MCP_CONFIG_FILE, .mcp.json, mcp.json)")
    p.add_argument("--no-mcp", action="store_true", help="Disable MCP server discovery for this run")
    p.add_argument("--skills-dir", help="Directory containing skills/<name>/SKILL.md manifests")
    p.add_argument("--no-skills", action="store_true", help="Disable workspace skill discovery for this run")
    p.add_argument("--memory-dir", help="Directory for persistent memory records (default: <workspace>/.memory)")
    p.add_argument("--no-memory", action="store_true", help="Disable cross-session memory recall and extraction")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.demo:
        from .demo import run_demo
        raise SystemExit(run_demo())

    try:
        config = parse_config()
    except ConfigError as e:
        console.print(f"[red bold]Configuration error:[/red bold] {e}")
        sys.exit(1)

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.permission_mode:
        config.permission_mode = args.permission_mode
    if args.workspace_root:
        config.workspace_root = args.workspace_root
    if args.trace_dir:
        config.trace_dir = args.trace_dir
    if args.test_log_dir:
        config.test_log_dir = args.test_log_dir
    if args.tool_timeout is not None:
        config.tool_timeout_seconds = args.tool_timeout
    if args.max_retries is not None:
        config.max_retries = args.max_retries
    if args.sandbox:
        config.sandbox_backend = args.sandbox
    if args.docker_image:
        config.docker_image = args.docker_image
    if args.docker_network:
        config.docker_network_enabled = True
    if args.docker_cpus is not None:
        config.docker_cpus = args.docker_cpus
    if args.docker_memory:
        config.docker_memory = args.docker_memory
    if args.docker_pids_limit is not None:
        config.docker_pids_limit = args.docker_pids_limit
    if args.docker_writable_rootfs:
        config.docker_read_only_rootfs = False
    if args.docker_seccomp_profile:
        config.docker_seccomp_profile = args.docker_seccomp_profile
    # CLI显式参数优先级高于环境变量里的MCP配置。
    if args.mcp_config:
        config.mcp_config_file = args.mcp_config
    if args.no_mcp:
        config.mcp_enabled = False
    if args.skills_dir:
        config.skills_dir = args.skills_dir
    if args.no_skills:
        config.skills_enabled = False
    if args.memory_dir:
        config.memory_dir = args.memory_dir
    if args.no_memory:
        config.memory_enabled = False

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or ORBIT_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 ORBIT_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    try:
        harness = _build_harness(config)
    except ValueError as e:
        console.print(f"[red bold]Configuration error:[/red bold] {e}")
        sys.exit(1)
    try:
        agent = harness.create_agent(
            llm=llm,
            max_context_tokens=config.max_context_tokens,
            max_rounds=50,
            memory_enabled=config.memory_enabled,
        )

        # 记忆的召回/提取是隐藏的LLM调用，远程模型上可能耗时数十秒，输出进度避免终端像卡死。
        harness.set_memory_notify(lambda msg: console.print(f"[dim][memory] {msg}[/dim]"))

        # resume saved session
        if args.resume:
            if harness.resume_session(agent, args.resume, restore_model=not bool(args.model)):
                config.model = agent.llm.model
                console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
            else:
                console.print(f"[red]Session '{args.resume}' not found.[/red]")
                sys.exit(1)

        if args.prompt:
            _run_once(agent, args.prompt)
        else:
            _repl(agent, config)
    finally:
        trace_path = harness.close()
        console.print(f"[dim]Trace saved: {trace_path}[/dim]")


def _build_harness(config: Config) -> OrbitHarness:
    permission_mode = PermissionMode(config.permission_mode)
    workspace_root = Path(config.workspace_root).expanduser() if config.workspace_root else Path.cwd()
    trace_dir = Path(config.trace_dir).expanduser() if config.trace_dir else None
    test_log_dir = Path(config.test_log_dir).expanduser() if config.test_log_dir else None
    memory_dir = Path(config.memory_dir).expanduser() if config.memory_dir else None

    def approve(tool_name: str, arguments: dict, reason: str) -> bool:
        console.print(f"\n[yellow]Approval required:[/] {tool_name}")
        console.print(f"[dim]{reason}[/dim]")
        console.print(f"[dim]{_brief(arguments, maxlen=160)}[/dim]")
        reply = input("Approve? [y/N] ").strip().lower()
        return reply in {"y", "yes"}

    return OrbitHarness(
        HarnessConfig(
            workspace_root=workspace_root,
            trace_dir=trace_dir,
            test_log_dir=test_log_dir,
            permission_mode=permission_mode,
            tool_timeout_seconds=config.tool_timeout_seconds,
            max_retries=config.max_retries,
            sandbox_backend=config.sandbox_backend,
            docker_image=config.docker_image,
            docker_network_enabled=config.docker_network_enabled,
            docker_cpus=config.docker_cpus,
            docker_memory=config.docker_memory,
            docker_pids_limit=config.docker_pids_limit,
            docker_read_only_rootfs=config.docker_read_only_rootfs,
            docker_seccomp_profile=config.docker_seccomp_profile,
            memory_dir=memory_dir,
            memory_enabled=config.memory_enabled,
            mcp_enabled=config.mcp_enabled,
            mcp_config_file=config.mcp_config_file or None,
            skills_enabled=config.skills_enabled,
            skills_dir=_resolve_skills_dir(config),
        ),
        approval_callback=approve,
    )


def _resolve_skills_dir(config: Config) -> str | None:
    if not config.skills_enabled:
        return None
    if config.skills_dir:
        return str(Path(config.skills_dir).expanduser())
    workspace_root = Path(config.workspace_root).expanduser() if config.workspace_root else Path.cwd()
    # 默认优先读取workspace根目录skills，不存在时兼容包内orbit/skills。
    return str(default_skills_dir(workspace_root))

# _run_once() 是Orbit的非交互执行入口，负责跑一次用户prompt、实时打印模型输出、展示工具调用，并把中断或异常转换成清晰的终端退出行为。
def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        _print_tool_start(name, kwargs)

    def on_tool_result(name, kwargs, result):
        _print_tool_result(name, kwargs, result)

    try:
        agent.harness.run_chat(agent, prompt, on_token=on_token, on_tool=on_tool, on_tool_result=on_tool_result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        # one-shot mode: print whatever went wrong and exit non-zero
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    print()


def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    memory_line = (
        "\nMemory: [dim]disabled[/dim]"
        if not config.memory_enabled
        else f"\nMemory: [green]on[/green] [dim]({config.memory_dir or '<workspace>/.memory'})[/dim]"
    )
    console.print(Panel(
        f"[bold]Orbit[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + memory_line
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))
    # 给交互式命令行配置输入历史记录文件，把 ~ 展开成当前用户的home目录。
    hist_path = os.path.expanduser("~/.orbit_history")
    """
    # 创建一个prompt_toolkit历史记录对象。后面传给输入框：
    ```
    pt_prompt(
        "You > ",
        history=history,
        ...
    )
    ```
    这样你在Orbit交互式REPL里输入过的命令会被保存下来，下次可以用方向键上下翻历史输入。
    """
    history = FileHistory(hist_path)

    # 创建快捷键绑定对象，用来覆盖prompt_toolkit默认的回车行为。
    kb = KeyBindings()

    # 按Enter时提交当前输入，让REPL开始处理这一轮用户消息。
    @kb.add("enter")
    def _submit(event):
        # 先校验输入内容，再触发prompt_toolkit的提交处理。
        event.current_buffer.validate_and_handle()

    # 按Esc+Enter时插入换行，方便用户粘贴多行代码或多行需求。
    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    # 进入交互式REPL主循环，每一轮读取一次用户输入并处理。
    while True:
        try:
            # 读取终端输入；开启multiline、历史记录和自定义快捷键。
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        # 用户按Ctrl+D或Ctrl+C时退出REPL。
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        # 空输入不交给Agent处理，直接等待下一轮输入。
        if not user_input:
            continue

        # 处理内置命令，避免这些控制命令被当成普通prompt发给模型。
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        # 展示帮助面板。
        if user_input == "/help":
            _show_help()
            continue
        # 展示当前workspace可用skills，帮助用户知道模型能按需加载哪些专用规范。
        if user_input == "/skills":
            _show_skills(agent)
            continue
        # 清空当前对话历史，但保留同一个Agent实例和模型配置。
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        # 展示当前LLM实例累计的token消耗和可估算成本。
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            # 只有模型在价格表里时才展示成本估算。
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        # 查看或切换当前模型；/model无参数表示查看，有参数表示切换。
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            # 有新模型名时，同时更新LLM实例和运行期config。
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            # 没有新模型名时，只打印当前模型。
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        # 手动触发上下文压缩，并展示压缩前后的估算token数。
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.harness.compact(agent)
            after = estimate_tokens(agent.messages)
            # 如果真的发生压缩，展示压缩前后token变化。
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            # 如果未达到压缩条件，展示当前token和消息数量。
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        # 保存当前会话，后续可以通过orbit-r恢复。
        if user_input == "/save":
            sid = agent.harness.save_session(agent)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: orbit -r {sid}")
            continue
        # 展示本次会话通过edit_file/write_file记录到的变更文件。
        if user_input == "/diff":
            changed_files = agent.harness.changed_files()
            # 没有记录到文件变更时给出空状态提示。
            if not changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            # 有变更时按文件名排序输出。
            else:
                console.print(f"[bold]Files modified this session ({len(changed_files)}):[/bold]")
                for f in changed_files:
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        # 列出最近保存的会话，方便用户选择resume目标。
        if user_input == "/sessions":
            sessions = agent.harness.list_sessions()
            # 没有保存过会话时给出空状态提示。
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            # 有会话时展示id、模型、保存时间和首条用户消息预览。
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue
        # 展示跨会话记忆状态：记忆目录和已存储的记忆条目。
        if user_input == "/memory":
            _show_memory(agent)
            continue

        # 未知/命令不应该发给模型，直接提示用户查看/help。
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # 普通输入会进入Agent主循环。
        streamed: list[str] = []

        # 记录并实时打印模型流式返回的文本片段。
        def on_token(tok, streamed=streamed):
            streamed.append(tok)
            print(tok, end="", flush=True)

        # 工具执行前打印工具名和简略参数，给用户可观测性。
        def on_tool(name, kwargs):
            _print_tool_start(name, kwargs)

        # 工具执行后打印特殊状态。skills需要明确告诉用户是否已经加载成功。
        def on_tool_result(name, kwargs, result):
            _print_tool_result(name, kwargs, result)

        # 调用Harness顶层入口处理用户输入，内部可能经历多轮LLM调用和工具调用。
        # run_chat只在内存里追加trace事件；完整trace在用户退出REPL时由harness.close()统一落盘。
        try:
            response = _run_chat_interruptibly(
                agent,
                user_input,
                on_token=on_token,
                on_tool=on_tool,
                on_tool_result=on_tool_result,
            )
            # 如果已经流式打印过内容，这里只补一个换行。
            if streamed:
                print()
            # 如果没有流式内容，说明最终response是在工具调用后一次性返回的，用Markdown渲染。
            else:
                console.print(Markdown(response))
        # 当前轮被Ctrl+C中断时，不退出整个REPL，只提示中断。
        except (KeyboardInterrupt, ToolInterrupted):
            console.print(
                "\n[yellow]Interrupted. Add instructions below to continue the same task.[/yellow]"
            )
        # 其他异常也只打印错误，保证REPL还能继续接受下一轮输入。
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    # 打印交互式REPL支持的内置命令和输入快捷键。
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /skills        List available workspace skills\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  /memory        Show persistent memory store status\n"
        "  quit           Exit Orbit\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="Orbit Help",
        border_style="dim",
    ))


def _show_skills(agent: Agent) -> None:
    tool = _load_skill_tool(agent)
    if tool is None:
        console.print("[dim]Skill loading is disabled.[/dim]")
        return

    skills = agent.harness.list_skills(agent)
    if not skills:
        console.print("[dim]No workspace skills found.[/dim]")
        return

    console.print("[bold]Available skills:[/bold]")
    for index, skill in enumerate(skills, 1):
        name = escape(skill["name"])
        description = escape(skill["description"])
        console.print(f"  {index}. [cyan]{name}[/cyan]：{description}")


def _load_skill_tool(agent: Agent):
    return agent.harness.skill_tool(agent)


def _show_memory(agent: Agent) -> None:
    # 记忆被禁用（如--no-memory或子Agent）时给出明确提示。
    status = agent.harness.memory_status(agent)
    if status is None:
        console.print("[dim]Persistent memory is disabled.[/dim]")
        return
    records = status["records"]
    console.print(f"Memory dir: [cyan]{status['directory']}[/cyan]")
    if not records:
        console.print("[dim]No memories stored yet. Durable preferences and project facts are saved here automatically.[/dim]")
        return
    console.print(f"[bold]Stored memories ({len(records)}):[/bold]")
    for record in records:
        name = escape(record["name"])
        mem_type = escape(record["type"])
        description = escape(record["description"])
        console.print(f"  [cyan]{name}[/cyan] [dim]({mem_type})[/dim] - {description}")


def _print_tool_start(name: str, kwargs: dict) -> None:
    if name == "load_skill":
        skill_name = escape(str(kwargs.get("name") or ""))
        console.print(f"\n[green]Skill([bold]{skill_name}[/bold])[/green]")
        return
    console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")


def _print_tool_result(name: str, kwargs: dict, result: str) -> None:
    del kwargs
    if name != "load_skill":
        # 普通工具失败时也要显式告诉用户，否则像write_file缺少参数这种问题会看起来像卡住。
        if result.startswith(("Error:", "Blocked by harness")):
            console.print(f"  [red]└ {escape(result)}[/red]")
            return
        # write_file成功后打印落盘结果，用户能直接知道文件已经创建或覆盖。
        if name == "write_file":
            console.print(f"  [green]└ {escape(result)}[/green]")
        return
    if result.startswith(("Error:", "Blocked by harness")):
        console.print(f"  [red]└ Failed to load skill[/red] [dim]{escape(result)}[/dim]")
        return
    console.print("  [green]└ Successfully loaded skill[/green]")


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    if "content" in kwargs:
        kwargs = {
            **kwargs,
            "content": f"<{len(str(kwargs['content']))} chars>",
        }
    # 把工具参数压缩成单行短文本，避免工具调用提示占满终端。
    s = ", ".join(f"{k}={_brief_value(v)}" for k, v in kwargs.items())
    # 超过maxlen时截断并追加省略号。
    return s[:maxlen] + ("..." if len(s) > maxlen else "")


def _brief_value(value) -> str:
    return repr(value).replace("\n", "\\n")[:40]
