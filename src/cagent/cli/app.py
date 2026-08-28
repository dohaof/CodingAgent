"""The command-line entry point.

Three ways to run: one task and exit, an interactive session, or replaying a
trace from a previous run. All three build the same :class:`~cagent.agent.Agent`
and differ only in what drives it and what renders it.

Credentials are deliberately not accepted as a flag. A key on the command line
lands in shell history and in the process list, so it comes from an untracked
config file, and the failure message says so.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..agent.approval import ApprovalPolicy
from ..agent.engine import Agent
from ..agent.events import FanOutSink
from ..agent.trace import TraceWriter, read_trace
from ..config import AgentConfig, load_config
from ..errors import CagentError, ConfigError
from ..tools.registry import default_registry
from .render import ConsoleRenderer, prompt_for_approval

__all__ = ["main"]

_BANNER_HELP = """\
Commands: /help  /tools  /cost  /context  /approve <mode>  /clear  /trace  /exit
Ctrl-C interrupts the current task; Ctrl-C again at the prompt exits."""


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="cagent",
        description="A coding agent that reads, edits, and runs code to finish a task.",
        epilog=(
            "An endpoint is three settings — base_url, model, and api_key — set "
            "in .cagent.toml (or ~/.cagent.toml). The key is never taken from a "
            "flag, because a flag lands in shell history and the process list."
        ),
    )
    parser.add_argument(
        "task", nargs="*", help="the task to perform; omit for an interactive session"
    )

    endpoint = parser.add_argument_group("endpoint")
    endpoint.add_argument(
        "--base-url",
        metavar="URL",
        help="the API endpoint, e.g. https://api.example.com/v1 (used verbatim)",
    )
    endpoint.add_argument(
        "--model", metavar="NAME", help="the model name your endpoint serves"
    )
    endpoint.add_argument(
        "--wire",
        choices=("openai", "anthropic"),
        help="request format: openai (Chat Completions, the default) or anthropic (Messages)",
    )
    endpoint.add_argument(
        "--no-key",
        action="store_true",
        help="the endpoint needs no API key, e.g. a local Ollama or llama.cpp server",
    )
    endpoint.add_argument("--temperature", type=float, help="sampling temperature")

    limits = parser.add_argument_group("limits")
    limits.add_argument("--max-steps", type=int, help="model requests allowed per task")
    limits.add_argument("--token-budget", type=int, help="stop after this many tokens")
    limits.add_argument("--context-window", type=int, help="model's context window, for compaction")
    limits.add_argument("--bash-timeout", type=float, help="seconds before a command is killed")

    safety = parser.add_argument_group("safety")
    safety.add_argument(
        "--approval",
        choices=("suggest", "auto-edit", "full-auto"),
        help="suggest: confirm every change; auto-edit: confirm commands only; "
        "full-auto: confirm only destructive commands",
    )
    safety.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="shorthand for --approval full-auto",
    )
    safety.add_argument("--workspace", type=Path, help="project directory (default: current)")
    safety.add_argument(
        "--allow-outside-workspace",
        action="store_true",
        help="permit reads and writes outside the workspace",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--no-repo-map", action="store_true", help="omit the project map")
    output.add_argument("--no-thinking", action="store_true", help="hide the reasoning trace")
    output.add_argument("--quiet", action="store_true", help="only warnings and the summary")
    output.add_argument("--trace-dir", type=Path, help="write a JSONL trace here")
    output.add_argument("--no-trace", action="store_true", help="disable tracing")

    info = parser.add_argument_group("information")
    info.add_argument("--show-config", action="store_true", help="print the resolved configuration")
    info.add_argument("--list-tools", action="store_true", help="print the tools and their schemas")
    info.add_argument("--replay", type=Path, metavar="TRACE", help="replay a trace file")
    info.add_argument("--version", action="version", version=f"cagent {__version__}")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    """Map parsed flags onto config field names.

    ``None`` entries are dropped by the loader, so an unspecified flag leaves
    the config files in charge.
    """
    approval = "full-auto" if args.yes else args.approval
    values: dict[str, object] = {
        "model": args.model,
        "base_url": args.base_url,
        "wire": args.wire,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "token_budget": args.token_budget,
        "context_window": args.context_window,
        "bash_timeout": args.bash_timeout,
        "approval_mode": approval,
        "workspace": args.workspace,
    }
    if args.no_key:
        values["requires_key"] = False
    if args.allow_outside_workspace:
        values["allow_outside_workspace"] = True
    if args.no_repo_map:
        values["repo_map_enabled"] = False
    if args.trace_dir is not None:
        values["trace_dir"] = args.trace_dir
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(highlight=False, soft_wrap=False)

    if args.replay is not None:
        return _replay(console, args.replay)

    try:
        # An explicit workspace is also the project-config root.  This makes
        # ``cagent --workspace D:\\project`` behave like starting from inside
        # that project, even when the shell remains in another directory.
        config = load_config(_overrides(args), cwd=args.workspace)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

    if args.no_trace:
        config.trace_dir = None
    elif config.trace_dir is None:
        config.trace_dir = config.workspace / ".cagent" / "traces"

    if args.show_config:
        return _show_config(console, config)
    if args.list_tools:
        return _list_tools(console)

    try:
        config.validate()
    except ConfigError as exc:
        # validate() names the specific missing setting, so this only shows the
        # shape of a complete configuration.
        console.print(f"[red]Configuration error:[/red] {exc}")
        console.print(
            "[dim]A complete endpoint needs three things, in .cagent.toml:\n"
            # Escaped: rich would read an unknown [cagent] tag as markup.
            "  \\[cagent]\n"
            '  base_url = "https://api.example.com/v1"\n'
            '  model = "<a model that endpoint serves>"\n'
            '  api_key = "<your key>"\n'
            'Add wire = "anthropic" for the Messages API, or --no-key for a '
            "local server. Run 'cagent --show-config' to see what resolved.[/dim]"
        )
        return 2

    task = " ".join(args.task).strip()
    return _run_session(console, config, task, args)


def _run_session(
    console: Console,
    config: AgentConfig,
    task: str,
    args: argparse.Namespace,
) -> int:
    """Build the agent and drive it, one task or interactively."""
    renderer = ConsoleRenderer(
        config,
        console=console,
        quiet=args.quiet,
        show_thinking=not args.no_thinking,
    )
    sink = FanOutSink([renderer])

    interactive = not task
    policy = ApprovalPolicy(
        config,
        prompter=(lambda request: prompt_for_approval(console, request))
        if sys.stdin.isatty()
        else None,
    )

    try:
        agent = Agent.create(config, sink=sink, policy=policy, registry=default_registry())
    except CagentError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        return 2

    trace = TraceWriter.create(config, session_id=agent.session_id)
    if trace is not None:
        sink.sinks.append(trace)
        if trace.error:
            console.print(f"[yellow]![/yellow] tracing disabled: {trace.error}")

    _install_interrupt_handler(agent, console)
    agent.announce(task or "(interactive)")

    exit_code = 0
    try:
        if interactive:
            exit_code = _repl(console, agent, config)
        else:
            result = agent.run_turn(task)
            exit_code = 0 if result.completed else 1
    finally:
        agent.finish(
            "finished" if exit_code == 0 else "incomplete",
            trace_path=str(trace.path) if trace and not trace.error else None,
        )
        if trace is not None:
            trace.close()
        agent.close()
        if sink.failures:
            console.print(f"[dim]display/trace sinks dropped: {'; '.join(sink.failures)}[/dim]")
    return exit_code


def _install_interrupt_handler(agent: Agent, console: Console) -> None:
    """Make Ctrl-C interrupt the task rather than kill the process.

    A run that is killed outright loses the summary and the trace's closing
    record, so the first interrupt asks the loop to stop at its next safe point
    and the second is left to Python's default behaviour.
    """
    interrupted = {"count": 0}

    def handler(signum: int, frame: FrameType | None) -> None:
        interrupted["count"] += 1
        if interrupted["count"] == 1:
            agent.interrupt()
            console.print("\n[yellow]Interrupting… (Ctrl-C again to force quit)[/yellow]")
            return
        raise KeyboardInterrupt

    # Not on the main thread, or a platform without SIGINT control: the default
    # handler stays, which is a degraded but working session.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, handler)


def _repl(console: Console, agent: Agent, config: AgentConfig) -> int:
    """Read tasks from the user until they leave."""
    console.print(Panel(_BANNER_HELP, border_style="dim", expand=False))
    while True:
        try:
            console.print("\n[bold cyan]›[/bold cyan] ", end="")
            line = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _command(console, agent, config, text):
                return 0
            continue

        agent.reset_interrupt()
        result = agent.run_turn(text)
        if not result.completed:
            console.print(f"[dim](turn ended: {result.stopped_by})[/dim]")


def _command(console: Console, agent: Agent, config: AgentConfig, line: str) -> bool:
    """Handle a slash command. Returns True when the session should end."""
    parts = line.split()
    name, arguments = parts[0], parts[1:]

    match name:
        case "/exit" | "/quit":
            return True
        case "/help":
            console.print(Panel(_BANNER_HELP, border_style="dim", expand=False))
        case "/tools":
            _list_tools(console)
        case "/cost":
            usage = agent.usage
            console.print(
                f"prompt {usage.prompt_tokens:,} · completion {usage.completion_tokens:,} "
                f"· cached {usage.cached_tokens:,} · {agent.guard.steps} steps"
            )
        case "/context":
            tokens = agent.context.token_count()
            console.print(
                f"{tokens:,} of {config.context_window:,} tokens "
                f"({agent.context.pressure:.0%}) · {len(agent.context.history)} messages "
                f"· {agent.context.compactions} compaction(s)"
            )
        case "/clear":
            agent.context.history.clear()
            agent.guard.note_progress()
            console.print("[dim]history cleared[/dim]")
        case "/approve":
            if arguments and arguments[0] in ("suggest", "auto-edit", "full-auto"):
                config.approval_mode = arguments[0]  # type: ignore[assignment]
                console.print(f"[dim]approval mode: {arguments[0]}[/dim]")
            else:
                console.print(f"approval mode: {config.approval_mode}")
        case "/trace":
            console.print(str(config.trace_dir or "tracing disabled"))
        case _:
            console.print(f"[dim]unknown command {name}; try /help[/dim]")
    return False


def _show_config(console: Console, config: AgentConfig) -> int:
    """Print the resolved configuration, with the key masked.

    Reports a missing endpoint or model as "not set" rather than raising: this
    command is most useful precisely when the configuration is incomplete.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    if config.api_key:
        key_state = "set"
    elif not config.requires_key:
        key_state = "not needed (--no-key)"
    else:
        key_state = "[red]not set[/red] (api_key in .cagent.toml)"

    rows = {
        "base_url": config.base_url or "[red]not set[/red] (base_url in .cagent.toml)",
        "model": config.model or "[red]not set[/red] (model in .cagent.toml)",
        "wire": config.wire,
        "api key": key_state,
        "workspace": str(config.workspace),
        "approval mode": config.approval_mode,
        "context window": f"{config.context_window:,}",
        "compact at": f"{config.compact_at_tokens:,} tokens",
        "max steps": str(config.max_steps),
        "token budget": str(config.token_budget or "unlimited"),
        "bash timeout": f"{config.bash_timeout:g}s",
        "fuzzy threshold": f"{config.fuzzy_threshold:.2f}",
        "repo map": f"{config.repo_map_token_budget} tokens" if config.repo_map_enabled else "off",
        "trace dir": str(config.trace_dir or "off"),
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(Panel(table, title="resolved configuration", border_style="cyan", expand=False))

    if config.base_url and config.model:
        console.print(
            f"[dim]requests go to "
            f"{config.resolved_base_url}"
            f"{'/messages' if config.wire == 'anthropic' else '/chat/completions'}[/dim]"
        )
    return 0


def _list_tools(console: Console) -> int:
    """Print each tool with the schema the model actually receives."""
    registry = default_registry()
    table = Table(box=None, padding=(0, 2))
    table.add_column("tool", style="bold")
    table.add_column("risk")
    table.add_column("arguments", style="dim")
    for tool in sorted(registry, key=lambda t: t.name):
        schema = tool.spec().input_schema
        properties = schema.get("properties")
        required = schema.get("required")
        names = list(properties) if isinstance(properties, dict) else []
        mandatory = set(required) if isinstance(required, list) else set()
        # A trailing "?" marks an optional argument, which is the one thing a
        # reader of this table actually needs from the schema.
        rendered = ", ".join(f"{name}{'' if name in mandatory else '?'}" for name in names)
        table.add_row(tool.name, tool.risk.name.lower(), rendered)
    console.print(table)
    return 0


def _replay(console: Console, path: Path) -> int:
    """Re-narrate a recorded run.

    Useful because the interesting question about a finished session is usually
    "what did it actually do", and the trace answers it without a second run
    that would behave differently anyway.
    """
    try:
        records = read_trace(path)
    except OSError as exc:
        console.print(f"[red]Could not read {path}:[/red] {exc}")
        return 2
    if not records:
        console.print(f"[yellow]{path} contains no events.[/yellow]")
        return 1

    for record in records:
        kind = record.get("type")
        stamp = f"[dim]{record.get('t', 0):7.2f}s[/dim]"
        match kind:
            case "session":
                # Older traces recorded a "provider" name; fall back to it so a
                # trace written before that field was dropped still replays.
                where = record.get("endpoint") or record.get("provider") or "unknown endpoint"
                console.print(
                    Panel(
                        f"{record.get('model')} @ {where} · "
                        f"{record.get('workspace')} · {record.get('approval_mode')}",
                        title="replay",
                        border_style="cyan",
                        expand=False,
                    )
                )
            case "user":
                console.print(f"{stamp} [bold cyan]›[/bold cyan] {record.get('text', '')}")
            case "step_started":
                console.print(
                    f"{stamp} [dim]· step {record.get('step')} "
                    f"({record.get('prompt_tokens_estimate', 0):,} tokens)[/dim]"
                )
            case "step_finished":
                usage = record.get("usage", {})
                text = _first_text(record.get("message", {}))
                console.print(
                    f"{stamp} [dim]{record.get('finish_reason')} · "
                    f"{usage.get('prompt', 0)}+{usage.get('completion', 0)} tokens · "
                    f"{record.get('latency_s', 0):.2f}s[/dim]"
                )
                if text:
                    console.print(f"        {text}")
            case "tool_started":
                console.print(
                    f"{stamp} ⏺ [bold]{record.get('name')}[/bold] "
                    f"[dim]{_short(record.get('arguments'))}[/dim]"
                )
            case "tool_finished":
                mark = "[red]✗[/red]" if record.get("is_error") else "[green]✓[/green]"
                first = str(record.get("content", "")).strip().split("\n", 1)[0]
                console.print(f"        ⎿ {mark} {first[:100]}")
            case "approval_requested":
                console.print(f"{stamp} [yellow]?[/yellow] {record.get('summary')}")
            case "approval_decided":
                verdict = "approved" if record.get("approved") else "declined"
                if not record.get("automatic"):
                    console.print(f"        [dim]{verdict}[/dim]")
            case "compaction":
                console.print(
                    f"{stamp} [dim]· compacted ({record.get('strategy')}): "
                    f"{record.get('tokens_before', 0):,} → {record.get('tokens_after', 0):,}[/dim]"
                )
            case "warning":
                console.print(f"{stamp} [yellow]![/yellow] {record.get('message')}")
            case "run_finished":
                usage = record.get("usage", {})
                console.print(
                    Panel(
                        f"{record.get('reason')} · {record.get('steps')} steps · "
                        f"{record.get('elapsed_s', 0):.1f}s · "
                        f"{usage.get('prompt', 0):,}+{usage.get('completion', 0):,} tokens",
                        border_style="cyan",
                        expand=False,
                    )
                )
    return 0


def _first_text(message: dict[str, object]) -> str:
    """The prose of a recorded message, if it had any."""
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", "")).strip()
    return ""


def _short(value: object, limit: int = 90) -> str:
    """Render recorded arguments compactly."""
    if not isinstance(value, dict):
        return ""
    rendered = ", ".join(f"{key}={str(item)[:40]}" for key, item in value.items())
    return rendered[:limit]


def _entry() -> NoReturn:
    """Console-script wrapper: translate exceptions into an exit code."""
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    _entry()
