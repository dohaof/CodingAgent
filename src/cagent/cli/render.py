"""Turning engine events into something worth watching.

This is the only module that prints. It implements
:class:`~cagent.agent.events.EventSink`, so the engine stays unaware of it and
the same run can be rendered here, recorded to a trace, and asserted on in a
test at once.

Two rendering decisions shape the rest:

*Prose streams, structure does not.* Assistant text is shown as it arrives,
re-rendered as Markdown in place, because waiting for a complete paragraph feels
like a hang. Tool activity is shown only once it has happened, as a settled line
with its result, because a half-executed command is not information.

*The diff is the receipt.* When a tool changed a file the diff is printed, and
that is deliberately the only place file contents appear — it means the user can
follow what changed without the model spending tokens narrating it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..agent.approval import Decision
from ..agent.events import (
    AgentEvent,
    ApprovalDecided,
    ApprovalRequested,
    CompactionDone,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    UserMessage,
    Warning,
)
from ..config import AgentConfig
from ..tools.base import ApprovalRequest
from ..types import RiskLevel, ToolCallPart
from .pricing import estimate_cost, parse_prices

__all__ = ["ConsoleRenderer", "prompt_for_approval"]

_RISK_STYLE = {
    RiskLevel.SAFE: "green",
    RiskLevel.MUTATING: "yellow",
    RiskLevel.DANGEROUS: "bold red",
}

_MAX_ARG_CHARS = 68
_MAX_RESULT_LINES = 12
_MAX_DIFF_LINES = 40


def _format_arguments(call: ToolCallPart) -> str:
    """Condense a call's arguments to one readable line.

    The interesting argument is almost always a path or a command, so those are
    shown bare and everything else is shown as ``key=value``, with long values
    clipped. The full arguments are in the trace when they are needed.
    """
    if not call.arguments:
        return ""
    parts: list[str] = []
    for key, value in call.arguments.items():
        text = value.replace("\n", "⏎") if isinstance(value, str) else str(value)
        if len(text) > _MAX_ARG_CHARS:
            text = text[:_MAX_ARG_CHARS] + "…"
        parts.append(text if key in ("path", "command", "pattern") else f"{key}={text}")
    return ", ".join(parts)


@dataclass(slots=True)
class ConsoleRenderer:
    """Renders a session to a terminal.

    ``quiet`` suppresses the streamed prose and per-tool lines, leaving warnings
    and the closing summary. The evaluation harness runs with it set, since a
    hundred sessions of streamed output is noise, not observability.
    """

    config: AgentConfig
    console: Console = field(default_factory=lambda: Console(highlight=False, soft_wrap=False))
    quiet: bool = False
    show_thinking: bool = True

    _live: Live | None = None
    _buffer: str = ""
    _thinking: str = ""
    _step: int = 0
    _tools_run: int = 0
    _files_touched: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ sink

    def handle(self, event: AgentEvent) -> None:
        """Render one event. Never raises: a display fault must not stop a run."""
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001  # rendering is never load-bearing
            self._stop_live()
            self.console.print(f"[dim](display error: {type(exc).__name__}: {exc})[/dim]")

    def _dispatch(self, event: AgentEvent) -> None:
        match event:
            case RunStarted():
                self._on_run_started(event)
            case UserMessage():
                pass  # the user just typed it; echoing is noise
            case StepStarted():
                self._on_step_started(event)
            case ThinkingDelta():
                self._on_thinking(event)
            case TextDelta():
                self._on_text(event)
            case StepFinished():
                self._on_step_finished(event)
            case ApprovalRequested():
                self._stop_live()
            case ApprovalDecided():
                self._on_approval_decided(event)
            case ToolStarted():
                self._on_tool_started(event)
            case ToolFinished():
                self._on_tool_finished(event)
            case CompactionDone():
                self._on_compaction(event)
            case Warning():
                self._on_warning(event)
            case TurnFinished():
                self._stop_live()
            case RunFinished():
                self._on_run_finished(event)

    # --------------------------------------------------------------- sections

    def _on_run_started(self, event: RunStarted) -> None:
        if self.quiet:
            return
        header = Table.grid(padding=(0, 1))
        header.add_column(style="dim")
        header.add_column()
        header.add_row("model", event.model)
        header.add_row("endpoint", event.endpoint)
        header.add_row("workspace", str(self.config.workspace))
        header.add_row("approval", self.config.approval_mode)
        header.add_row(
            "sandbox",
            f"{self.config.sandbox_mode} ({self.config.sandbox_sync})",
        )
        header.add_row("tools", f"{len(event.tool_names)} · {', '.join(event.tool_names)}")
        header.add_row("system", f"{event.system_tokens} tokens")
        self.console.print(Panel(header, title="cagent", border_style="cyan", expand=False))

    def _on_step_started(self, event: StepStarted) -> None:
        self._step = event.step
        self._buffer = ""
        self._thinking = ""
        if self.quiet:
            return
        pressure = event.prompt_tokens_estimate / max(self.config.context_window, 1)
        self.console.print(
            f"[dim]· step {event.step}/{event.max_steps} · "
            f"{event.prompt_tokens_estimate:,} tokens ({pressure:.0%} of window)[/dim]"
        )
        self._start_live()

    def _on_thinking(self, event: ThinkingDelta) -> None:
        if self.quiet or not self.show_thinking:
            return
        self._thinking += event.text
        self._update_live()

    def _on_text(self, event: TextDelta) -> None:
        if self.quiet:
            return
        self._buffer += event.text
        self._update_live()

    def _on_step_finished(self, event: StepFinished) -> None:
        self._stop_live()
        if self.quiet:
            return
        if self._buffer.strip():
            # Printed once more as settled Markdown: the live region renders
            # partial syntax, and a half-closed code fence looks broken.
            self.console.print(Markdown(self._buffer.strip()))
        self._buffer = ""
        self._thinking = ""

    def _on_tool_started(self, event: ToolStarted) -> None:
        if self.quiet:
            return
        style = _RISK_STYLE.get(event.risk, "white")
        arguments = _format_arguments(event.call)
        line = Text("⏺ ", style=style)
        line.append(event.call.name, style=f"bold {style}")
        if arguments:
            line.append(f"({arguments})", style="dim")
        self.console.print(line)

    def _on_tool_finished(self, event: ToolFinished) -> None:
        self._tools_run += 1
        path = event.outcome.metadata.get("path")
        if isinstance(path, str):
            self._files_touched.add(path)
        if self.quiet:
            return

        marker = "[red]✗[/red]" if event.outcome.is_error else "[green]✓[/green]"
        summary = self._summarise_outcome(event)
        self.console.print(f"  ⎿ {marker} {summary} [dim]({event.duration_s:.2f}s)[/dim]")

        if event.outcome.display:
            self._print_display(event.outcome.display)
        elif event.outcome.is_error:
            self._print_result_body(event.outcome.content, style="red")
        else:
            self._print_result_body(event.outcome.content, style="dim")

    @staticmethod
    def _summarise_outcome(event: ToolFinished) -> str:
        """One line describing what a finished call achieved."""
        metadata = event.outcome.metadata
        if "exit_code" in metadata:
            code = metadata["exit_code"]
            if metadata.get("timeout"):
                return "timed out"
            return f"exit {code}"
        if "matches" in metadata:
            return f"{metadata['matches']} match(es)"
        if "lines_shown" in metadata:
            return f"{metadata['lines_shown']} of {metadata.get('total_lines', '?')} lines"
        if "added" in metadata or "removed" in metadata:
            return f"+{metadata.get('added', 0)}/-{metadata.get('removed', 0)}"
        first = event.outcome.content.strip().split("\n", 1)[0]
        return first[:80] if first else "done"

    def _print_display(self, display: str) -> None:
        """Render a tool's rich output — in practice, a unified diff."""
        lines = display.split("\n")
        clipped = lines[:_MAX_DIFF_LINES]
        body = "\n".join(clipped)
        if len(lines) > _MAX_DIFF_LINES:
            body += f"\n… {len(lines) - _MAX_DIFF_LINES} more diff lines"
        if display.startswith(("--- ", "diff ")):
            self.console.print(
                Syntax(body, "diff", theme="ansi_dark", background_color="default")
            )
        else:
            self.console.print(Text(body, style="dim"))

    def _print_result_body(self, content: str, *, style: str) -> None:
        """Show a bounded excerpt of a tool result, indented under its call."""
        text = content.strip()
        if not text:
            return
        lines = text.split("\n")
        shown = lines[:_MAX_RESULT_LINES]
        for line in shown:
            self.console.print(Text("    " + line[:200], style=style))
        if len(lines) > _MAX_RESULT_LINES:
            self.console.print(
                Text(f"    … {len(lines) - _MAX_RESULT_LINES} more lines", style="dim")
            )

    def _on_approval_decided(self, event: ApprovalDecided) -> None:
        if self.quiet or event.automatic:
            return
        verdict = "approved" if event.approved else "declined"
        note = " (remembered)" if event.remembered else ""
        style = "green" if event.approved else "red"
        self.console.print(f"[{style}]{verdict}{note}[/{style}]")

    def _on_compaction(self, event: CompactionDone) -> None:
        if self.quiet:
            return
        self.console.print(
            f"[dim]· compacted history ({event.strategy}): "
            f"{event.tokens_before:,} → {event.tokens_after:,} tokens, "
            f"{event.messages_before} → {event.messages_after} messages[/dim]"
        )

    def _on_warning(self, event: Warning) -> None:
        self._stop_live()
        self.console.print(f"[yellow]![/yellow] {event.message}")
        if event.detail:
            self.console.print(f"[dim]  {event.detail}[/dim]")

    def _on_run_finished(self, event: RunFinished) -> None:
        self._stop_live()
        usage = event.usage
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("steps", str(event.steps))
        table.add_row("tools run", str(self._tools_run))
        table.add_row("elapsed", f"{event.elapsed_s:.1f}s")
        table.add_row("prompt tokens", f"{usage.prompt_tokens:,}")
        table.add_row("completion tokens", f"{usage.completion_tokens:,}")
        if usage.cached_tokens:
            table.add_row("cached prompt tokens", f"{usage.cached_tokens:,}")
        if usage.reasoning_tokens:
            table.add_row("reasoning tokens", f"{usage.reasoning_tokens:,}")

        cost = estimate_cost(
            self.config.model_for_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            prices=parse_prices(self.config.prices),
        )
        if cost is not None:
            table.add_row("estimated cost", f"${cost:.4f}")
        else:
            # No rate configured. Reporting tokens and saying so beats printing
            # a number from a table that went stale after release.
            table.add_row("cost", "no rate set for this model")
        if self._files_touched:
            table.add_row("files touched", ", ".join(sorted(self._files_touched)[:6]))
        if event.trace_path:
            table.add_row("trace", event.trace_path)

        self.console.print(
            Panel(table, title=f"session {event.reason}", border_style="cyan", expand=False)
        )

    # ------------------------------------------------------------------- live

    def _start_live(self) -> None:
        """Open the streaming region for one model response."""
        if self._live is not None or not self.console.is_terminal:
            return
        self._live = Live(
            Text("thinking…", style="dim"),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def _update_live(self) -> None:
        """Redraw the streaming region with what has arrived so far."""
        if self._live is None:
            # Not a terminal (piped output, or a test): fall back to appending
            # raw deltas, which keeps piped logs readable.
            return
        blocks: list[Text | Markdown] = []
        if self._thinking.strip():
            blocks.append(Text(self._tail(self._thinking, 6), style="dim italic"))
        if self._buffer.strip():
            blocks.append(Markdown(self._buffer))
        self._live.update(Group(*blocks) if blocks else Text("thinking…", style="dim"))

    @staticmethod
    def _tail(text: str, lines: int) -> str:
        """The last few lines, for a reasoning trace that only scrolls."""
        wrapped = text.strip().split("\n")
        return "\n".join(wrapped[-lines:])

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()


def prompt_for_approval(console: Console, request: ApprovalRequest) -> Decision:
    """Ask the user about one call and return their decision.

    Renders the tool's own account of the action — for an edit, the real diff of
    what would be written — because a prompt the user cannot evaluate trains them
    to approve everything, which is worse than no prompt at all.

    ``always`` is offered only below the dangerous tier, matching the policy: the
    signature of a dangerous command is its full text, so remembering it could
    not be reused anyway.
    """
    style = _RISK_STYLE.get(request.risk, "yellow")
    body: list[Text | Syntax] = [Text(request.summary, style="bold")]
    if request.detail:
        detail = request.detail
        lines = detail.split("\n")
        if len(lines) > _MAX_DIFF_LINES:
            hidden = len(lines) - _MAX_DIFF_LINES
            detail = "\n".join(lines[:_MAX_DIFF_LINES]) + f"\n… {hidden} more lines"
        if detail.startswith("--- "):
            body.append(Syntax(detail, "diff", theme="ansi_dark", background_color="default"))
        else:
            body.append(Text(detail))

    console.print(
        Panel(
            Group(*body),
            title=f"{request.risk.name.lower()} · {request.tool}",
            border_style=style,
            expand=False,
        )
    )

    allow_always = request.risk is not RiskLevel.DANGEROUS and not request.always_prompt
    # Escape the brackets: Rich treats ``[y]`` as a markup tag otherwise and
    # the user sees ``es o lways uit`` with the shortcut letters swallowed.
    options = r"\[y]es  \[n]o" + (r"  \[a]lways" if allow_always else "") + r"  \[q]uit"
    while True:
        console.print(f"[{style}]{options}[/{style}] ", end="")
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return Decision(approved=False, abort=True)

        if answer in ("y", "yes", ""):
            return Decision(approved=True)
        if answer in ("n", "no"):
            return Decision(approved=False)
        if answer in ("a", "always") and allow_always:
            return Decision(approved=True, remember=True)
        if answer in ("q", "quit"):
            return Decision(approved=False, abort=True)
        console.print("[dim]Please answer y, n, a, or q.[/dim]")


def terminal_width() -> int:
    """Usable console width, with a sane default for a non-tty."""
    return shutil.get_terminal_size((100, 24)).columns


def open_console(stream: TextIO | None = None) -> Console:
    """A console configured the way this app wants it."""
    return Console(file=stream, highlight=False, soft_wrap=False)
