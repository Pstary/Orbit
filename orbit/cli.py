"""Interactive REPL - the user-facing terminal interface."""

import argparse
import os
import sys
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import Agent
from .config import Config, ConfigError, parse_config
from .harness import OrbitHarness, HarnessConfig, PermissionMode
from .llm import LLM, LiteLLM
from .session import list_sessions, load_session, save_session

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="orbit",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $ORBIT_MODEL or gpt-5.5)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
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
    agent = harness.create_agent(
        llm=llm,
        max_context_tokens=config.max_context_tokens,
        max_rounds=50,
    )

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        try:
            _run_once(agent, args.prompt)
        finally:
            trace_path = harness.close()
            console.print(f"[dim]Trace saved: {trace_path}[/dim]")
        return

    # interactive REPL
    try:
        _repl(agent, config)
    finally:
        trace_path = harness.close()
        console.print(f"[dim]Trace saved: {trace_path}[/dim]")


def _build_harness(config: Config) -> OrbitHarness:
    permission_mode = PermissionMode(config.permission_mode)
    workspace_root = Path(config.workspace_root).expanduser() if config.workspace_root else Path.cwd()
    trace_dir = Path(config.trace_dir).expanduser() if config.trace_dir else None
    test_log_dir = Path(config.test_log_dir).expanduser() if config.test_log_dir else None

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
        ),
        approval_callback=approve,
    )

# _run_once() 是Orbit的非交互执行入口，负责跑一次用户prompt、实时打印模型输出、展示工具调用，并把中断或异常转换成清晰的终端退出行为。
def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    try:
        agent.harness.run_chat(agent, prompt, on_token=on_token, on_tool=on_tool)
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
    console.print(Panel(
        f"[bold]Orbit[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))
    # 给交互式命令行配置输入历史记录文件，把 ~ 展开成当前用户的home目录。
    hist_path = os.path.expanduser("~/.orbit_history")
    """
    # 创建一个 prompt_toolkit 的历史记录对象。后面传给输入框：
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
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
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
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: orbit -r {sid}")
            continue
        # 展示本次会话通过edit_file/write_file记录到的变更文件。
        if user_input == "/diff":
            from .tools.edit import _changed_files
            # 没有记录到文件变更时给出空状态提示。
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            # 有变更时按文件名排序输出。
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        # 列出最近保存的会话，方便用户选择resume目标。
        if user_input == "/sessions":
            sessions = list_sessions()
            # 没有保存过会话时给出空状态提示。
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            # 有会话时展示id、模型、保存时间和首条用户消息预览。
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
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
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        # 调用Harness顶层入口处理用户输入，内部可能经历多轮LLM调用和工具调用。
        # run_chat只在内存里追加trace事件；完整trace在用户退出REPL时由harness.close()统一落盘。
        try:
            response = agent.harness.run_chat(agent, user_input, on_token=on_token, on_tool=on_tool)
            # 如果已经流式打印过内容，这里只补一个换行。
            if streamed:
                print()
            # 如果没有流式内容，说明最终response是在工具调用后一次性返回的，用Markdown渲染。
            else:
                console.print(Markdown(response))
        # 当前轮被Ctrl+C中断时，不退出整个REPL，只提示中断。
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        # 其他异常也只打印错误，保证REPL还能继续接受下一轮输入。
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    # 打印交互式REPL支持的内置命令和输入快捷键。
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit Orbit\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="Orbit Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    # 把工具参数压缩成单行短文本，避免工具调用提示占满终端。
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    # 超过maxlen时截断并追加省略号。
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
