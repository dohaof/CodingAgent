"""The command-line entry point.

The command-line surface starts either one task or an interactive session.
Interactive commands can inspect the agent, control the sandbox, and restore a
conversation from a trace.

Credentials are deliberately not accepted as a flag. A key on the command line
lands in shell history and in the process list, so it comes from an untracked
config file, and the failure message says so.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..agent.approval import ApprovalPolicy
from ..agent.engine import Agent
from ..agent.events import FanOutSink
from ..agent.trace import TraceWriter, history_from_trace, read_trace
from ..config import AgentConfig, load_config
from ..errors import CagentError, ConfigError
from ..tools.registry import default_registry
from ..types import Message
from .render import ConsoleRenderer, prompt_for_approval, render_restored_history
from .resume import TraceChoice as _TraceChoice
from .resume import find_trace_choices as _find_trace_choices
from .resume import first_user_prompt as _first_user_prompt
from .resume import resolve_trace_reference as _resolve_trace_reference
from .resume import resume_trace_dir as _resume_trace_dir

__all__ = ["main"]

_BANNER_HELP = """\
Commands: /help <instruct>  /tools  /cost  /context  /approve <mode>
          /sandbox  /resume [ID|PATH]  /undo  /clear  /exit
Use /help <instruct> for details about one command.
Ctrl+C interrupts active work or exits while idle.
Full-screen TTY: selected idle text is copied; Ctrl+R resumes; F1 helps;
Ctrl+Q exits."""

_RESUME_HELP = """\
Conversation recovery

  /resume             list saved conversations and choose one
  /resume ID          restore a saved conversation by its short ID
  /resume PATH        restore a conversation from an explicit JSONL path

With no argument, traces are read from the configured trace directory (by
default <workspace>/.cagent/traces) and shown with their time, first request,
step count, and last known status. The current Agent, configuration, API key,
workspace, approvals, and sandbox
remain active. The trace is read-only. It restores recorded user, assistant,
thinking, tool-call, and tool-result messages, then the next normal prompt
continues from that history. It does not restore an old Docker container,
unsynchronised sandbox files, token usage, or clipped tool output. A trace from
another project is accepted as conversation context, but the current workspace
is used for tools and a warning is shown. Only traces with at least one
non-empty user request and provider-valid history appear in the picker; startup
metadata or interrupted traces with no restorable user turn are hidden."""

_UNDO_HELP = """\
Undo the latest conversation turn

  /undo

Removes the most recent user message and every assistant/tool message that
answered it. Tool calls and results are removed as one unit so the remaining
history stays valid for the model. This changes conversation context only: it
does not revert files, commands, installed packages, or other side effects."""

_SANDBOX_HELP = """\
Docker sandbox - one disposable container per Agent conversation

By default `sandbox_mode = "auto"` checks Docker Desktop/Engine and the selected
local image. When both exist, cagent creates an isolated project snapshot and
uses one container for this conversation. It never pulls images automatically.
If either check fails, it falls back to the host and prints a warning.

  /sandbox                         show status, image, and sync policy
  /sandbox on [IMAGE]              copy the project and enable Docker isolation
  /sandbox image IMAGE             select an image (safe while sandbox is on)
  /sandbox sync never|ask|always   choose what /sandbox off or exit does
  /sandbox apply                   sync changes now; keep sandbox enabled
  /sandbox rollback                discard unsynced changes; keep sandbox enabled
  /sandbox off                     apply the sync policy, then return to host mode

Inside the sandbox, file tools and run_bash use the temporary snapshot. The
real project is unchanged until /sandbox apply, or until /sandbox off/exit
syncs it according to the selected policy. The normal workspace path boundary
and command approvals still apply. The container starts on the first shell
command, is reused for this conversation, and is removed when it closes.

`/sandbox off` explicitly selects the host shell. A host process can escape its
initial workspace with cd, absolute paths, redirection, symlinks, or children;
approval and risk classification are not an isolation boundary.

`allow_outside_workspace = false` remains the default and confines file tools,
so host fallback normally shows `PATHS workspace-only / SHELL host
(unrestricted)`. Set it true only to allow file tools outside the workspace.

Examples:
  /sandbox image my-project-agent:latest
  /sandbox sync ask
  /sandbox on
  ... work and test ...
  /sandbox apply       (keep the sandbox and continue)
  /sandbox off         (finish and return to the real project)"""


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
        help="grant unrestricted file and host-shell access outside the workspace",
    )
    safety.add_argument(
        "--sandbox",
        choices=("auto", "off", "docker"),
        help="run shell commands in a disposable Docker snapshot "
        "when available (default: auto; host fallback is unrestricted)",
    )
    safety.add_argument(
        "--sandbox-sync",
        choices=("never", "ask", "always"),
        help="after Docker runs: discard, ask before syncing, or sync automatically",
    )
    safety.add_argument(
        "--sandbox-image",
        metavar="IMAGE",
        help="local Docker image for the sandbox (default: python:3.12-slim)",
    )
    safety.add_argument("--sandbox-memory-mb", type=int, help="Docker memory limit in MiB")
    safety.add_argument("--sandbox-cpus", type=float, help="Docker CPU limit")
    safety.add_argument("--sandbox-pids", type=int, help="Docker process limit")
    safety.add_argument(
        "--sandbox-workspace-mb",
        type=int,
        help="maximum regular-file bytes retained in the disposable workspace",
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
        "token_budget": args.token_budget,
        "context_window": args.context_window,
        "bash_timeout": args.bash_timeout,
        "approval_mode": approval,
        "workspace": args.workspace,
        "sandbox_mode": args.sandbox,
        "sandbox_sync": args.sandbox_sync,
        "sandbox_image": args.sandbox_image,
        "sandbox_memory_mb": args.sandbox_memory_mb,
        "sandbox_cpus": args.sandbox_cpus,
        "sandbox_pids": args.sandbox_pids,
        "sandbox_workspace_mb": args.sandbox_workspace_mb,
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
    if interactive and sys.stdin.isatty() and sys.stdout.isatty() and not args.quiet:
        try:
            from .tui import run_tui_session
        except ModuleNotFoundError as exc:
            if exc.name != "textual":
                raise
            console.print(
                "[red]Interactive TUI dependency is missing:[/red] textual\n"
                "[dim]Install the project again in this virtual environment with "
                "'.\\venv\\Scripts\\python.exe -m pip install -e .' "
                "or 'python -m pip install textual'.[/dim]"
            )
            return 2

        try:
            return run_tui_session(
                config,
                quiet=args.quiet,
                show_thinking=not args.no_thinking,
            )
        except CagentError as exc:
            console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
            return 2
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

    interrupt_state = _install_interrupt_handler(agent, console)
    agent.announce(task or "(interactive)")

    exit_code = 0
    try:
        if interactive:
            exit_code = _repl(console, agent, config, interrupt_state)
        else:
            interrupt_state.busy = True
            try:
                result = agent.run_turn(task)
            finally:
                interrupt_state.busy = False
            exit_code = 0 if result.completed else 1
    finally:
        trace_path = (
            str(trace.path)
            if trace is not None and trace.has_user_message and not trace.error
            else None
        )
        reason = "interrupted" if agent.abort.is_set() else (
            "finished" if exit_code == 0 else "incomplete"
        )
        agent.finish(reason, trace_path=trace_path)
        if trace is not None:
            trace.discard_if_empty()
        agent.close()
        if sink.failures:
            console.print(f"[dim]display/trace sinks dropped: {'; '.join(sink.failures)}[/dim]")
    return exit_code


@dataclass(slots=True)
class _InterruptState:
    """Whether the foreground agent turn is currently running."""

    busy: bool = False


def _install_interrupt_handler(agent: Agent, console: Console) -> _InterruptState:
    """Interrupt an active turn; let Ctrl-C at an idle prompt exit."""
    state = _InterruptState()

    def handler(signum: int, frame: FrameType | None) -> None:
        if state.busy:
            agent.interrupt()
            console.print("\n[yellow]Interrupting current turn...[/yellow]")
            return
        raise KeyboardInterrupt

    # Not on the main thread, or a platform without SIGINT control: the default
    # handler stays, which is a degraded but working session.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, handler)
    return state


def _repl(
    console: Console,
    agent: Agent,
    config: AgentConfig,
    interrupt_state: _InterruptState | None = None,
) -> int:
    """Read tasks from the user until they leave."""
    state = interrupt_state or _InterruptState()
    console.print(Panel(Text(_BANNER_HELP), border_style="dim", expand=False))
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
        state.busy = True
        try:
            result = agent.run_turn(text)
        finally:
            state.busy = False
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
            _print_help(console, arguments[0] if arguments else None)
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
        case "/undo":
            removed = agent.undo_last_turn()
            if not removed:
                console.print("[dim]nothing to undo[/dim]")
            else:
                _record_history_checkpoint(agent)
                console.print(
                    f"[dim]removed the latest user turn ({removed} message(s)); "
                    "files and command effects were not reverted[/dim]"
                )
        case "/sandbox":
            _sandbox_command(console, agent, arguments)
        case "/resume":
            _resume_command(console, agent, config, arguments)
        case "/approve":
            if arguments and arguments[0] in ("suggest", "auto-edit", "full-auto"):
                config.approval_mode = arguments[0]  # type: ignore[assignment]
                console.print(f"[dim]approval mode: {arguments[0]}[/dim]")
            else:
                console.print(f"approval mode: {config.approval_mode}")
        case _:
            console.print(f"[dim]unknown command {name}; try /help[/dim]")
    return False


_HELP_TOPICS = {
    "help": (
        "Help",
        "Use /help for the command list, or /help <instruct> for one command's details.\n\n"
        "In a terminal, Ctrl+C interrupts active work or exits while idle; an idle "
        "selection is copied instead. Ctrl+R resumes, F1 opens this help, and Ctrl+Q exits.",
        "dim",
    ),
    "resume": ("Resume", _RESUME_HELP, "green"),
    "undo": ("Undo", _UNDO_HELP, "yellow"),
    "sandbox": ("Sandbox", _SANDBOX_HELP, "cyan"),
    "tools": (
        "Tools",
        "List the tools and argument shapes currently available to the agent.",
        "dim",
    ),
    "cost": (
        "Cost",
        "Show token usage and the number of model steps used in this session.",
        "dim",
    ),
    "context": (
        "Context",
        "Show estimated context usage, retained messages, and compaction count.",
        "dim",
    ),
    "approve": (
        "Approve",
        "Use /approve suggest, /approve auto-edit, or /approve full-auto to change\n"
        "the approval policy for later tool calls.",
        "dim",
    ),
    "clear": (
        "Clear",
        "Discard the current conversation history. This cannot restore unsaved context.",
        "dim",
    ),
    "exit": (
        "Exit",
        "Leave the interactive session. Sandbox changes follow the selected sync policy.",
        "dim",
    ),
}


def _print_help(console: Console, instruction: str | None = None) -> None:
    """Render the overview or details for one interactive instruction."""
    if instruction is None:
        console.print(
            Panel(Text(_BANNER_HELP), title="Commands", border_style="dim", expand=False)
        )
        return

    topic = instruction.lower().lstrip("/")
    detail = _HELP_TOPICS.get(topic)
    if detail is None:
        console.print(
            f"[yellow]No detailed help for /{topic}.[/yellow] "
            "Try /help resume or /help sandbox."
        )
        return
    title, body, border = detail
    console.print(Panel(Text(body), title=title, border_style=border, expand=False))


def _resume_command(
    console: Console,
    agent: Agent,
    config: AgentConfig,
    arguments: list[str],
) -> None:
    """Restore a trace into the current interactive Agent.

    The normal path is a numbered picker. A short session ID or an explicit
    path remains available for scripts and for users who already know which
    conversation they need.
    """
    path_text = " ".join(arguments).strip().strip('"')
    trace_dir = _resume_trace_dir(config)
    choices = _find_trace_choices(trace_dir)

    if not path_text:
        path = _pick_trace(console, choices, trace_dir)
        if path is None:
            return
    else:
        path = _resolve_trace_reference(path_text, trace_dir)
        if path is None and path_text.isdecimal() and choices:
            index = int(path_text) - 1
            if 0 <= index < len(choices):
                path = choices[index].path
        if path is None:
            console.print(
                f"[red]resume:[/red] no trace named {path_text!r} in {trace_dir}\n"
                "[dim]Use /resume to list saved conversations, or provide a full path.[/dim]"
            )
            return

    try:
        records = read_trace(path)
    except OSError as exc:
        console.print(f"[red]resume:[/red] could not read {path}: {exc}")
        return
    if not records:
        console.print(f"[yellow]resume: {path} contains no events.[/yellow]")
        return

    if _first_user_prompt(records) is None:
        console.print(
            f"[yellow]resume: {path} has no non-empty user request and cannot be restored.[/yellow]"
        )
        return

    history = history_from_trace(records)
    if not history:
        console.print(f"[yellow]resume: {path} has no restorable conversation history.[/yellow]")
        return

    recorded_workspace = next(
        (record.get("workspace") for record in records if record.get("type") == "session"),
        None,
    )
    if (
        isinstance(recorded_workspace, str)
        and Path(recorded_workspace).expanduser().resolve() != config.workspace
    ):
        console.print(
            f"[yellow]resume: trace workspace is {recorded_workspace}; "
            f"current workspace is {config.workspace}.[/yellow]"
        )

    agent.restore_history(history)
    _record_history_checkpoint(agent, history)
    render_restored_history(console, history, session_id=path.stem)
    console.print(f"[dim]resumed {len(history)} message(s) from {path}[/dim]")


def _record_history_checkpoint(
    agent: Agent, messages: Sequence[Message] | None = None
) -> None:
    """Persist context replacement in the active append-only trace, if any."""
    sink = getattr(agent, "sink", None)
    candidates = (sink, *getattr(sink, "sinks", ()))
    for candidate in candidates:
        if isinstance(candidate, TraceWriter):
            history = messages if messages is not None else agent.context.history
            candidate.record_history(history)
            return


def _pick_trace(
    console: Console,
    choices: list[_TraceChoice],
    trace_dir: Path,
) -> Path | None:
    """Render the trace picker and ask for a one-based choice."""
    if not choices:
        trace_count = len(list(trace_dir.glob("*.jsonl"))) if trace_dir.is_dir() else 0
        if trace_count:
            console.print(
                f"[yellow]resume: found {trace_count} trace file(s) in {trace_dir}, "
                "but none contain a non-empty user turn that can be restored.[/yellow]"
            )
            return None
        console.print(
            f"[yellow]resume: no saved conversations in {trace_dir}.[/yellow]\n"
            "[dim]Run an interactive task first; traces are created automatically.[/dim]"
        )
        return None

    trace_count = len(list(trace_dir.glob("*.jsonl"))) if trace_dir.is_dir() else len(choices)
    if trace_count > len(choices):
        console.print(
            f"[dim]Showing {len(choices)} restorable conversation(s) out of "
            f"{trace_count} trace file(s); empty or incomplete traces are hidden.[/dim]"
        )
    table = Table(title=f"Saved conversations · {trace_dir}", header_style="cyan")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("started", width=17)
    table.add_column("id", width=14)
    table.add_column("steps", justify="right", width=6)
    table.add_column("status", width=12)
    table.add_column("task / request")
    for index, choice in enumerate(choices, start=1):
        started = dt.datetime.fromtimestamp(choice.modified).astimezone().strftime("%Y-%m-%d %H:%M")
        prompt = " ".join(choice.prompt.split())
        if len(prompt) > 58:
            prompt = prompt[:55] + "..."
        table.add_row(
            str(index),
            started,
            choice.session_id[:14],
            str(choice.steps),
            choice.status,
            prompt,
        )
    console.print(table)
    try:
        answer = input("Resume number (Enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        console.print("[dim]resume cancelled[/dim]")
        return None
    if not answer:
        console.print("[dim]resume cancelled[/dim]")
        return None
    if not answer.isdecimal() or not 1 <= int(answer) <= len(choices):
        console.print(f"[yellow]resume: choose a number from 1 to {len(choices)}.[/yellow]")
        return None
    return choices[int(answer) - 1].path


def _sandbox_command(console: Console, agent: Agent, arguments: list[str]) -> None:
    """Inspect or change the Docker sandbox between interactive turns."""
    action = arguments[0].lower() if arguments else "status"
    try:
        if action in ("status", "show"):
            path_policy = (
                "unrestricted"
                if agent.config.allow_outside_workspace and agent.sandbox is None
                else "workspace-only"
            )
            shell_policy = (
                "container"
                if agent.sandbox is not None
                else "host (unrestricted)"
            )
            console.print(
                f"sandbox: {agent.sandbox_status()}\n"
                f"paths: {path_policy}\n"
                f"shell: {shell_policy}\n"
                f"image: {agent.config.sandbox_image}\n"
                f"sync: {agent.config.sandbox_sync}"
            )
        elif action == "on":
            image = arguments[1] if len(arguments) > 1 else None
            agent.enable_sandbox(image=image)
            console.print(f"[dim]sandbox enabled: {agent.sandbox_status()}[/dim]")
        elif action == "off":
            agent.disable_sandbox()
            console.print(
                "[bold yellow]WARNING:[/bold yellow] sandbox disabled; "
                "run_bash now uses the unrestricted host"
            )
        elif action == "apply" and len(arguments) == 1:
            applied = agent.apply_sandbox_changes()
            if applied:
                console.print(f"[dim]sandbox changes applied: {len(applied)}[/dim]")
        elif action in ("rollback", "discard") and len(arguments) == 1:
            agent.discard_sandbox_changes()
            console.print("[dim]sandbox changes discarded; sandbox remains enabled[/dim]")
        elif action == "image" and len(arguments) == 2:
            image = arguments[1].strip()
            if not image:
                raise ValueError("image must not be empty")
            agent.set_sandbox_image(image)
            console.print(f"[dim]sandbox image: {image}[/dim]")
        elif action == "sync" and len(arguments) == 2 and arguments[1] in (
            "never",
            "ask",
            "always",
        ):
            agent.config.sandbox_sync = arguments[1]  # type: ignore[assignment]
            console.print(f"[dim]sandbox sync: {arguments[1]}[/dim]")
        else:
            console.print(
                "[dim]usage: /sandbox [status|on [IMAGE]|apply|rollback|off|"
                "image IMAGE|sync never|ask|always][/dim]"
            )
    except (CagentError, ValueError) as exc:
        console.print(f"[red]sandbox:[/red] {exc}")


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

    shell_execution = (
        "host (unrestricted)"
        if config.allow_outside_workspace or config.sandbox_mode == "off"
        else "auto (container if available; host (unrestricted) fallback)"
        if config.sandbox_mode == "auto"
        else "container"
    )
    rows = {
        "base_url": config.base_url or "[red]not set[/red] (base_url in .cagent.toml)",
        "model": config.model or "[red]not set[/red] (model in .cagent.toml)",
        "wire": config.wire,
        "api key": key_state,
        "workspace": str(config.workspace),
        "approval mode": config.approval_mode,
        "sandbox": f"{config.sandbox_mode} ({config.sandbox_sync})",
        "path boundary": "unrestricted" if config.allow_outside_workspace else "workspace-only",
        "shell execution": shell_execution,
        "context window": f"{config.context_window:,}",
        "compact at": f"{config.compact_at_tokens:,} tokens",
        "token budget": str(config.token_budget or "unlimited"),
        "bash timeout": f"{config.bash_timeout:g}s",
        "fuzzy threshold": f"{config.fuzzy_threshold:.2f}",
        "repo map": f"{config.repo_map_token_budget} tokens" if config.repo_map_enabled else "off",
        "trace dir": str(config.trace_dir or "off"),
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(Panel(table, title="resolved configuration", border_style="cyan", expand=False))

    if config.sandbox_mode == "auto" and not config.allow_outside_workspace:
        console.print(
            "[bold yellow]WARNING:[/bold yellow] Docker/image availability is checked "
            "when the Agent starts; auto mode falls back to host (unrestricted) if "
            "either prerequisite is missing."
        )
    elif config.sandbox_mode == "off":
        console.print(
            "[bold yellow]WARNING:[/bold yellow] sandbox is off; run_bash uses the "
            "host with unrestricted process access."
        )

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


def _entry() -> NoReturn:
    """Console-script wrapper: translate exceptions into an exit code."""
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    _entry()
