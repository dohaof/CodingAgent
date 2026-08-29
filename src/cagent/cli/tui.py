"""Full-screen interactive terminal UI.

The agent loop remains synchronous and framework-independent. Textual owns only
the presentation layer: model work runs in a worker thread and typed agent
events are posted back to the UI message queue.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, Static, TextArea

from ..agent.approval import ApprovalPolicy, Decision
from ..agent.engine import Agent, TurnResult
from ..agent.events import (
    AgentEvent,
    ApprovalDecided,
    ApprovalRequested,
    CompactionDone,
    FanOutSink,
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
from ..agent.trace import TraceWriter, history_from_trace, read_trace
from ..config import AgentConfig
from ..tools.base import ApprovalRequest
from ..tools.registry import default_registry
from ..types import Message, RiskLevel
from .render import restored_history_renderables
from .resume import (
    TraceChoice,
    find_trace_choices,
    first_user_prompt,
    resolve_trace_reference,
    resume_trace_dir,
)

__all__ = ["CagentTui", "run_tui_session"]

_MAX_DETAIL_LINES = 16


class AgentEventArrived(TextualMessage):
    """An engine event forwarded safely from its worker thread."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class TurnCompleted(TextualMessage):
    def __init__(self, result: TurnResult | None, error: str | None = None) -> None:
        super().__init__()
        self.result = result
        self.error = error


class CommandCompleted(TextualMessage):
    def __init__(self, output: str, error: str | None = None) -> None:
        super().__init__()
        self.output = output
        self.error = error


class SessionClosed(TextualMessage):
    def __init__(self, error: str | None = None) -> None:
        super().__init__()
        self.error = error


@dataclass(slots=True)
class TuiEventSink:
    app: CagentTui

    def handle(self, event: AgentEvent) -> None:
        self.app.post_message(AgentEventArrived(event))


@dataclass(slots=True)
class _ApprovalWaiter:
    ready: threading.Event = field(default_factory=threading.Event)
    decision: Decision = field(default_factory=lambda: Decision(False, abort=True))

    def resolve(self, decision: Decision | None) -> None:
        if self.ready.is_set():
            return
        self.decision = decision or Decision(False, abort=True)
        self.ready.set()


class ResumeListItem(ListItem):
    def __init__(self, choice: TraceChoice, number: int) -> None:
        self.choice = choice
        timestamp = dt.datetime.fromtimestamp(choice.modified).astimezone().strftime(
            "%Y-%m-%d %H:%M"
        )
        prompt = " ".join(choice.prompt.split())
        if len(prompt) > 90:
            prompt = prompt[:87] + "..."
        label = Text.assemble(
            (f"{number:>2}  ", "dim"),
            (f"{timestamp}  ", "cyan"),
            (f"{choice.session_id[:14]:<14}  ", "green"),
            (f"{choice.steps:>3} steps  ", "dim"),
            (f"{choice.status:<11}  ", "yellow"),
            prompt,
        )
        super().__init__(Label(label))


class ResumeScreen(ModalScreen[Path | None]):
    """Keyboard-driven saved-conversation picker."""

    CSS = """
    ResumeScreen {
        align: center middle;
        background: rgba(8, 11, 12, 0.78);
    }
    #resume-dialog {
        width: 95%;
        max-width: 130;
        height: 88%;
        max-height: 34;
        padding: 1 2;
        border: round #4fa88a;
        background: #151a1b;
    }
    #resume-title {
        height: 2;
        color: #83d6b8;
        text-style: bold;
    }
    #resume-list {
        height: 1fr;
        background: #101415;
    }
    #resume-list ListItem {
        height: auto;
        min-height: 2;
        padding: 0 1;
    }
    #resume-list ListItem.--highlight {
        background: #28473e;
    }
    #resume-actions {
        height: 3;
        align-horizontal: right;
        padding-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, choices: list[TraceChoice]) -> None:
        super().__init__()
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="resume-dialog"):
            yield Static("Resume conversation", id="resume-title")
            yield ListView(
                *(ResumeListItem(choice, index) for index, choice in enumerate(self.choices, 1)),
                id="resume-list",
            )
            with Horizontal(id="resume-actions"):
                yield Button("Cancel [Esc]", id="cancel")

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ResumeListItem):
            self.dismiss(item.choice.path)

    @on(Button.Pressed, "#cancel")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)


class ComposerInput(Input):
    """Input whose hint is hidden while an IME may be composing text."""

    def __init__(self, *, prompt_hint: str, id: str) -> None:
        self._prompt_hint = prompt_hint
        super().__init__(placeholder=prompt_hint, id=id)

    def set_prompt_hint(self, prompt_hint: str) -> None:
        """Update the hint without showing it under focused IME pre-edit text."""
        self._prompt_hint = prompt_hint
        self.placeholder = "" if self.has_focus else prompt_hint

    def _on_focus(self, event: events.Focus) -> None:
        super()._on_focus(event)
        self.placeholder = ""

    def _on_blur(self, event: events.Blur) -> None:
        super()._on_blur(event)
        self.placeholder = self._prompt_hint


class TranscriptTextArea(TextArea):
    """Selectable transcript with lightweight semantic line highlighting."""

    # Keep the transcript selectable with the mouse, but never make it the
    # keyboard target. In particular, this prevents IMEs from activating when
    # the user clicks or drags in the read-only conversation history.
    can_focus = False
    _right_button_selecting = False

    def _watch_selection(self, previous_selection: Any, selection: Any) -> None:
        """Keep mouse selection from moving the terminal's input cursor.

        Textual's ``TextArea`` watcher updates ``app.cursor_position`` for every
        selection, even when the widget is read-only and not focusable. Restore
        the position owned by the actual focused input after the base watcher
        performs its scrolling and selection bookkeeping.
        """
        if not self.is_mounted:
            super()._watch_selection(previous_selection, selection)
            return
        cursor_position = self.app.cursor_position
        super()._watch_selection(previous_selection, selection)
        self.app.cursor_position = cursor_position

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """Allow selection without showing an editable text caret."""
        if event.button == 3:
            if self.selected_text:
                self.action_copy()
                self.app.notify("Selected text copied", timeout=1.5)
                event.stop()
                event.prevent_default()
                return
            self._right_button_selecting = True
        self.screen.set_focus(None, scroll_visible=False)
        await super()._on_mouse_down(event)
        self._pause_blink(visible=False)

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        """Copy a selection completed with the right mouse button."""
        right_button_selecting = event.button == 3 and self._right_button_selecting
        await super()._on_mouse_up(event)
        self._right_button_selecting = False
        if right_button_selecting and self.selected_text:
            self.action_copy()
            self.app.notify("Selected text copied", timeout=1.5)
            event.stop()
            event.prevent_default()

    def _end_mouse_selection(self) -> None:
        """Finish selection while keeping the transcript caret hidden."""
        super()._end_mouse_selection()
        self._pause_blink(visible=False)

    def get_line(self, line_index: int) -> Text:
        line = super().get_line(line_index)
        plain = line.plain
        if plain in {"USER", "ASSISTANT", "SYSTEM"}:
            styles = {
                "USER": "bold cyan reverse",
                "ASSISTANT": "bold green reverse",
                "SYSTEM": "bold magenta reverse",
            }
            line.stylize(styles[plain])
        elif plain.startswith("TOOL ERROR:"):
            line.stylize("bold red reverse")
        elif plain.startswith("TOOL RESULT:"):
            line.stylize("bold green reverse")
        elif plain.startswith("TOOL:") or plain.startswith("  TOOL CALL:"):
            line.stylize("bold yellow reverse")
        elif plain.startswith("APPROVAL · DANGEROUS"):
            line.stylize("bold red reverse")
        elif plain.startswith("APPROVAL ·") or plain.startswith("[Y]ES"):
            line.stylize("bold yellow reverse")
        elif plain == "THINKING":
            line.stylize("dim italic")
        elif plain in {"COMMAND", "WORKING DIRECTORY", "PURPOSE"}:
            line.stylize("bold")
        elif plain.startswith("    "):
            stripped = plain.lstrip()
            number, separator, _ = stripped.partition("  ")
            if separator and number.isdigit():
                start = len(plain) - len(stripped)
                line.stylize("cyan", start, start + len(number))
        elif plain.startswith("+++") or (
            plain.startswith("+") and not plain.startswith("+++")
        ):
            line.stylize("green")
        elif plain.startswith("---") or (
            plain.startswith("-") and not plain.startswith("---")
        ):
            line.stylize("red")
        return line


class CagentTui(App[int]):
    """The full-screen interactive session."""

    TITLE = "cagent"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen {
        background: #0e1213;
        color: #dce3e3;
        layout: vertical;
    }
    #conversation {
        height: 1fr;
        padding: 1 2;
        background: #0e1213;
        border: none;
        scrollbar-color: #405457;
        scrollbar-background: #171d1e;
    }
    #conversation:focus {
        border: none;
    }
    #activity {
        height: 1;
        padding: 0 2;
        background: #1a2021;
        color: #92a2a4;
    }
    #composer {
        height: 3;
        border: tall #4f8f7a;
        background: #111617;
        color: #eef4f3;
        padding: 0 1;
    }
    #composer:focus {
        border: tall #79d2b1;
    }
    Footer {
        background: #1a2021;
        color: #aab7b8;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Copy / interrupt", priority=True),
        Binding("ctrl+q", "quit_session", "Quit", priority=True),
        Binding("ctrl+r", "resume", "Resume", priority=True),
        Binding("f1", "help", "Help"),
    ]

    def __init__(
        self,
        config: AgentConfig,
        *,
        quiet: bool = False,
        show_thinking: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quiet = quiet
        self.show_thinking = show_thinking
        self._busy = False
        self._closing_session = False
        self._closed_session = False
        self._exit_after_turn = False
        self._text_buffer = ""
        self._thinking_buffer = ""
        self._stream_kind: str | None = None
        self._stream_start_offset: int | None = None
        self._approval_waiter: _ApprovalWaiter | None = None
        self._approval_request: ApprovalRequest | None = None
        self._restored_from: str | None = None

        self.event_sink = TuiEventSink(self)
        self.sink = FanOutSink([self.event_sink])
        self.policy = ApprovalPolicy(config, prompter=self._prompt_for_approval)
        self.agent = Agent.create(
            config,
            sink=self.sink,
            policy=self.policy,
            registry=default_registry(),
        )
        self.trace = TraceWriter.create(config, session_id=self.agent.session_id)
        if self.trace is not None:
            self.sink.sinks.append(self.trace)

    def compose(self) -> ComposeResult:
        yield TranscriptTextArea(
            "",
            read_only=True,
            soft_wrap=True,
            show_line_numbers=False,
            id="conversation",
        )
        yield Static("Ready", id="activity")
        yield ComposerInput(prompt_hint="Ask cagent or enter /help", id="composer")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#composer", Input).focus()
        self.agent.announce("(interactive)")
        self._update_status()

    def on_unmount(self) -> None:
        if self._approval_waiter is not None:
            self._approval_waiter.resolve(Decision(False, abort=True))
            self._approval_waiter = None
            self._approval_request = None

    @on(Input.Submitted, "#composer")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if self._approval_waiter is not None:
            self._handle_approval_input(text)
            return
        if not text:
            return
        if self._busy:
            self._write(Text("The agent is busy. Press Ctrl+C to interrupt it.", style="yellow"))
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._start_turn(text)

    def _start_turn(self, text: str) -> None:
        self.agent.reset_interrupt()
        self._set_busy(True, "Working")
        self.run_worker(
            lambda: self._run_turn(text),
            name="agent-turn",
            group="agent",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _run_turn(self, text: str) -> None:
        try:
            result = self.agent.run_turn(text)
        except Exception as exc:  # noqa: BLE001  # preserve the TUI after an engine fault
            self.post_message(TurnCompleted(None, f"{type(exc).__name__}: {exc}"))
        else:
            self.post_message(TurnCompleted(result))

    @on(TurnCompleted)
    def on_turn_completed(self, message: TurnCompleted) -> None:
        self._set_busy(False, "Ready")
        if message.error:
            self._write(Text(f"Agent error: {message.error}", style="red"))
        elif message.result is not None and not message.result.completed:
            self._write(Text(f"Turn ended: {message.result.stopped_by}", style="yellow"))
        self._update_status()
        if self._exit_after_turn:
            self._begin_close()

    @on(AgentEventArrived)
    def on_agent_event_arrived(self, message: AgentEventArrived) -> None:
        self._render_event(message.event)

    def _render_event(self, event: AgentEvent) -> None:
        match event:
            case RunStarted():
                self._write(self._session_panel(event))
                self._update_status()
            case UserMessage():
                self._write_label("USER", "cyan")
                self._write(Text(event.text))
            case StepStarted():
                pressure = event.prompt_tokens_estimate / max(self.config.context_window, 1)
                self._text_buffer = ""
                self._thinking_buffer = ""
                self._stream_kind = None
                self._stream_start_offset = None
                self._set_activity(
                    f"Working · step {event.step} · "
                    f"{event.prompt_tokens_estimate:,} tokens ({pressure:.0%})"
                )
            case ThinkingDelta():
                if self.show_thinking and not self.quiet:
                    self._thinking_buffer += event.text
                    self._append_stream_delta(event.text, kind="thinking")
            case TextDelta():
                if not self.quiet:
                    self._text_buffer += event.text
                    self._append_stream_delta(event.text, kind="assistant")
            case StepFinished():
                self._finish_stream(event.message)
            case ApprovalRequested():
                self._set_activity(f"Waiting for approval · {event.request.tool}")
            case ApprovalDecided():
                if not event.automatic:
                    verdict = "Approved" if event.approved else "Declined"
                    note = " for this session" if event.remembered else ""
                    style = "green" if event.approved else "red"
                    self._write(Text(verdict + note, style=style))
            case ToolStarted():
                if not self.quiet:
                    arguments = self._format_arguments(event.call.arguments)
                    suffix = f"({arguments})" if arguments else "()"
                    style = "red" if event.risk is RiskLevel.DANGEROUS else "yellow"
                    if event.call.name == "read_file":
                        suffix = ""
                    self._write_label(f"TOOL: {event.call.name.upper()}{suffix}", style)
            case ToolFinished():
                self._render_tool_result(event)
            case CompactionDone():
                self._write(
                    Text(
                        f"Compacted context ({event.strategy}): "
                        f"{event.tokens_before:,} -> {event.tokens_after:,} tokens",
                        style="yellow",
                    )
                )
            case Warning():
                self._write(Text(f"Warning: {event.message}", style="yellow"))
                if event.detail:
                    self._write(Text(event.detail, style="dim"))
            case TurnFinished():
                self._set_activity("Ready")
            case RunFinished():
                self._write(
                    Text(
                        f"Session {event.reason} · {event.steps} steps · "
                        f"{event.usage.total:,} tokens",
                        style="dim",
                    )
                )

    def _finish_stream(self, message: Message) -> None:
        text = message.text
        if text.strip() and not self.quiet:
            if self._stream_kind != "assistant":
                self._write_label("ASSISTANT", "green")
                self._append_transcript("\n" + text.strip())
            elif self._stream_start_offset is None:
                self._append_transcript("\n" + text.strip())
            elif text != self._text_buffer:
                # Provider deltas may be normalized by the final message. Keep
                # the live transcript authoritative unless the final text is
                # genuinely different, in which case replace just this reply.
                transcript = self.query_one("#conversation", TextArea)
                prefix = transcript.text[: self._stream_start_offset]
                transcript.load_text(prefix + text.strip())
        self._text_buffer = ""
        self._thinking_buffer = ""
        self._stream_kind = None
        self._stream_start_offset = None

    def _append_stream_delta(self, text: str, *, kind: str) -> None:
        """Append a provider delta directly to the selectable transcript."""
        if not text or self.quiet:
            return
        if self._stream_kind != kind:
            if kind == "thinking":
                self._write_label("THINKING", "dim italic")
            else:
                self._write_label("ASSISTANT", "green")
            self._stream_kind = kind
            self._append_transcript("\n")
            if kind == "assistant":
                self._stream_start_offset = len(self.query_one("#conversation", TextArea).text)
        self._append_transcript(text)

    def _append_transcript(self, text: str) -> None:
        transcript = self.query_one("#conversation", TextArea)
        transcript.insert(text, transcript.document.end)
        if self.focused is not transcript or not transcript.selected_text:
            transcript.scroll_end(animate=False)

    def _render_tool_result(self, event: ToolFinished) -> None:
        if self.quiet:
            return
        metadata = event.outcome.metadata
        if "exit_code" in metadata:
            summary = f"exit {metadata['exit_code']}"
        elif "matches" in metadata:
            summary = f"{metadata['matches']} match(es)"
        elif "lines_shown" in metadata:
            summary = f"{metadata['lines_shown']} of {metadata.get('total_lines', '?')} lines"
        elif "added" in metadata or "removed" in metadata:
            summary = f"+{metadata.get('added', 0)}/-{metadata.get('removed', 0)}"
        else:
            summary = event.outcome.content.strip().split("\n", 1)[0][:100] or "done"
        style = "red" if event.outcome.is_error else "green"
        label = "TOOL ERROR" if event.outcome.is_error else "TOOL RESULT"
        self._write(Text(f"{label}: {summary} ({event.duration_s:.2f}s)", style=style))
        detail = event.outcome.display or event.outcome.content
        if not detail:
            return
        lines = detail.splitlines()
        body = "\n".join(lines[:_MAX_DETAIL_LINES])
        if len(lines) > _MAX_DETAIL_LINES:
            body += f"\n[{len(lines) - _MAX_DETAIL_LINES} more lines]"
        path = metadata.get("path")
        if isinstance(path, str) and "lines_shown" in metadata:
            self._write(Text(path, style="bold cyan"))
            body = self._indent_numbered_lines(body)
        elif event.call.name == "grep_search":
            body = self._group_search_results(body)
        if detail.startswith(("--- ", "diff ")):
            self._write(Syntax(body, "diff", theme="ansi_dark", background_color="default"))
        else:
            self._write(Text(body, style="red" if event.outcome.is_error else "dim"))

    @staticmethod
    def _indent_numbered_lines(body: str) -> str:
        """Align numbered source lines beneath one file heading."""
        rendered: list[str] = []
        for line in body.splitlines():
            number, separator, text = line.lstrip().partition("\t")
            shown = (
                f"{number}  {text}"
                if separator and number.isdigit()
                else line.lstrip()
            )
            rendered.append("    " + shown)
        return "\n".join(rendered)

    @staticmethod
    def _group_search_results(body: str) -> str:
        """Show each grep result path once, followed by indented line numbers."""
        import re

        pattern = re.compile(r"^(.+):(\d+)([:-])\s(.*)$")
        grouped: list[str] = []
        current_path: str | None = None
        for line in body.splitlines():
            match = pattern.match(line)
            if match is None:
                grouped.append(line)
                continue
            path, number, separator, text = match.groups()
            if path != current_path:
                if grouped and grouped[-1] != "":
                    grouped.append("")
                grouped.append(path)
                current_path = path
            grouped.append(f"    {number}{separator} {text}")
        return "\n".join(grouped).strip("\n")

    @staticmethod
    def _format_arguments(arguments: dict[str, object]) -> str:
        values: list[str] = []
        for key, value in arguments.items():
            text = str(value).replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            values.append(text if key in ("path", "command", "pattern") else f"{key}={text}")
        return ", ".join(values)

    def _write_label(self, label: str, color: str) -> None:
        """Write a prominent semantic label with spacing for quick scanning."""
        transcript = self.query_one("#conversation", TextArea)
        if transcript.text:
            self._append_transcript("\n")
        self._write(Text(label.upper(), style=f"bold {color} reverse"))

    def _handle_command(self, line: str) -> None:
        parts = line.split()
        name, arguments = parts[0].lower(), parts[1:]
        if name in ("/exit", "/quit"):
            self._request_close()
        elif name == "/resume":
            self._open_resume(" ".join(arguments).strip().strip('"'))
        elif name == "/clear":
            self.agent.context.history.clear()
            self.agent.guard.note_progress()
            self.query_one("#conversation", TextArea).load_text("")
            self._write(Text("Conversation context cleared.", style="dim"))
            self._update_status()
        elif name == "/sandbox":
            self._run_line_command(line, background=True)
        else:
            self._run_line_command(line, background=False)

    def _run_line_command(self, line: str, *, background: bool) -> None:
        if background:
            self._set_busy(True, "Updating sandbox")
            self.run_worker(
                lambda: self._capture_line_command(line),
                name="slash-command",
                group="agent",
                thread=True,
                exclusive=True,
                exit_on_error=False,
            )
            return
        output, error = self._capture_line_command_text(line)
        if output.strip():
            self._write(Text(output.rstrip()))
        if error:
            self._write(Text(error, style="red"))
        self._update_status()

    def _capture_line_command(self, line: str) -> None:
        output, error = self._capture_line_command_text(line)
        self.post_message(CommandCompleted(output, error))

    def _capture_line_command_text(self, line: str) -> tuple[str, str | None]:
        from .app import _command

        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=100,
            highlight=False,
            soft_wrap=False,
            color_system=None,
        )
        try:
            _command(console, self.agent, self.config, line)
        except Exception as exc:  # noqa: BLE001  # slash commands must leave the TUI usable
            return buffer.getvalue(), f"{type(exc).__name__}: {exc}"
        return buffer.getvalue(), None

    @on(CommandCompleted)
    def on_command_completed(self, message: CommandCompleted) -> None:
        self._set_busy(False, "Ready")
        if message.output.strip():
            self._write(Text(message.output.rstrip()))
        if message.error:
            self._write(Text(message.error, style="red"))
        self._update_status()

    def _open_resume(self, reference: str = "") -> None:
        trace_dir = resume_trace_dir(self.config)
        choices = find_trace_choices(trace_dir)
        if reference:
            path = resolve_trace_reference(reference, trace_dir)
            if path is None and reference.isdecimal() and choices:
                index = int(reference) - 1
                path = choices[index].path if 0 <= index < len(choices) else None
            if path is None:
                self._write(
                    Text(f"No trace named {reference!r} in {trace_dir}.", style="yellow")
                )
                return
            self._restore_path(path)
            return

        if not choices:
            count = len(list(trace_dir.glob("*.jsonl"))) if trace_dir.is_dir() else 0
            message = (
                f"Found {count} trace file(s), but none contain a restorable user turn."
                if count
                else f"No saved conversations in {trace_dir}."
            )
            self._write(Text(message, style="yellow"))
            return
        self.push_screen(ResumeScreen(choices), self._resume_selected)

    def _resume_selected(self, path: Path | None) -> None:
        if path is not None:
            self._restore_path(path)
        self.query_one("#composer", Input).focus()

    def _restore_path(self, path: Path) -> None:
        try:
            records = read_trace(path)
        except OSError as exc:
            self._write(Text(f"Could not read {path}: {exc}", style="red"))
            return
        if first_user_prompt(records) is None:
            self._write(Text(f"{path.name} has no non-empty user request.", style="yellow"))
            return
        history = history_from_trace(records)
        if not history:
            self._write(Text(f"{path.name} has no restorable history.", style="yellow"))
            return

        recorded_workspace = next(
            (record.get("workspace") for record in records if record.get("type") == "session"),
            None,
        )
        workspace_warning: str | None = None
        if (
            isinstance(recorded_workspace, str)
            and Path(recorded_workspace).expanduser().resolve() != self.config.workspace
        ):
            workspace_warning = (
                f"Trace workspace was {recorded_workspace}; tools still use "
                f"{self.config.workspace}."
            )

        self.agent.restore_history(history)
        if self.trace is not None:
            self.trace.record_history(history)
        self._restored_from = path.stem
        self.query_one("#conversation", TextArea).load_text("")
        for renderable in restored_history_renderables(history, session_id=path.stem):
            self._write(renderable)
        if workspace_warning:
            self._write(Text(workspace_warning, style="yellow"))
        self._update_status()

    def _prompt_for_approval(self, request: ApprovalRequest) -> Decision:
        waiter = _ApprovalWaiter()
        self.call_from_thread(self._show_approval, request, waiter)
        waiter.ready.wait()
        return waiter.decision

    def _show_approval(self, request: ApprovalRequest, waiter: _ApprovalWaiter) -> None:
        self._approval_waiter = waiter
        self._approval_request = request
        risk = request.risk.name.upper()
        style = "red" if request.risk is RiskLevel.DANGEROUS else "yellow"
        self._write_label(f"APPROVAL · {risk} · {request.tool.upper()}", style)
        summary = request.summary
        if request.tool == "run_bash" and " — " in summary:
            summary = summary.rsplit(" — ", 1)[1]
            self._write(Text("PURPOSE", style="bold dim"))
            self._write(Text("  " + summary))
        elif request.tool != "run_bash":
            self._write(Text(summary, style="bold"))
        if request.detail:
            detail = request.detail
            lines = detail.splitlines()
            if len(lines) > _MAX_DETAIL_LINES:
                detail = "\n".join(lines[:_MAX_DETAIL_LINES]) + (
                    f"\n[{len(lines) - _MAX_DETAIL_LINES} more lines]"
                )
            if detail.startswith("--- "):
                self._write(
                    Syntax(detail, "diff", theme="ansi_dark", background_color="default")
                )
            else:
                detail_lines = detail.splitlines()
                command = next((line for line in detail_lines if line.startswith("$ ")), "")
                location = next((line for line in detail_lines if line.startswith("in ")), "")
                other = [line for line in detail_lines if line not in (command, location)]
                if command:
                    self._write(Text("COMMAND", style="bold dim"))
                    self._write(Text("  " + command, style="yellow"))
                if location:
                    self._write(Text("WORKING DIRECTORY", style="bold dim"))
                    self._write(Text("  " + location.removeprefix("in ")))
                if other:
                    self._write(Text("\n".join(other)))
        allow_always = request.risk is not RiskLevel.DANGEROUS and not request.always_prompt
        options = "[Y]ES  [N]O" + ("  [A]LWAYS" if allow_always else "") + "  [Q]UIT"
        self._write(Text(options, style=style))
        self._set_activity(f"Waiting for approval · {request.tool} · y/n/a/q")
        composer = self.query_one("#composer", ComposerInput)
        composer.set_prompt_hint("Approval: y/n/a/q (Enter = yes)")
        composer.focus()

    def _handle_approval_input(self, answer: str) -> None:
        """Resolve the inline y/n/a/q prompt without starting a new turn."""
        waiter = self._approval_waiter
        request = self._approval_request
        if waiter is None or request is None:
            return
        answer = answer.lower()
        allow_always = request.risk is not RiskLevel.DANGEROUS and not request.always_prompt
        decision: Decision | None = None
        if answer in ("", "y", "yes"):
            decision = Decision(True)
        elif answer in ("n", "no"):
            decision = Decision(False)
        elif answer in ("a", "always") and allow_always:
            decision = Decision(True, remember=True)
        elif answer in ("q", "quit"):
            decision = Decision(False, abort=True)
        else:
            self._write(Text("Please answer y, n, a, or q.", style="dim"))
            return

        style = "green" if decision.approved else "red"
        self._write(Text(f"> {answer or 'yes'}", style=style))
        self._approval_waiter = None
        self._approval_request = None
        waiter.resolve(decision)
        self.query_one("#composer", ComposerInput).set_prompt_hint(
            "Ask cagent or enter /help"
        )
        self._set_activity("Working")

    def action_interrupt(self) -> None:
        conversation = self.query_one("#conversation", TextArea)
        if conversation.selected_text and (self.focused is conversation or not self._busy):
            conversation.action_copy()
            self.notify("Selected text copied", timeout=1.5)
            return
        # Treat Ctrl+C as a session exit while a turn is running. The
        # cancellation is still graceful: the worker records its partial
        # response, then ``_finish_session`` flushes the trace.
        self._request_close()

    def action_quit_session(self) -> None:
        self._request_close()

    def action_resume(self) -> None:
        if self._busy:
            self._write(
                Text("Finish or interrupt the current turn before resuming.", style="yellow")
            )
        else:
            self._open_resume()

    def action_help(self) -> None:
        if not self._busy:
            self._run_line_command("/help", background=False)

    def _request_close(self) -> None:
        if self._closing_session:
            return
        if self._busy:
            self._exit_after_turn = True
            self._abort_pending_approval()
            self.agent.interrupt()
            self._set_activity("Stopping current work before exit")
            return
        self._begin_close()

    def _abort_pending_approval(self) -> None:
        waiter = self._approval_waiter
        if waiter is None:
            return
        waiter.resolve(Decision(False, abort=True))
        self._approval_waiter = None
        self._approval_request = None

    def _begin_close(self) -> None:
        if self._closing_session:
            return
        self._closing_session = True
        self._set_busy(True, "Closing session")
        self.run_worker(
            self._finish_session,
            name="close-session",
            group="agent",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _finish_session(self) -> None:
        error: str | None = None
        try:
            trace_path = (
                str(self.trace.path)
                if self.trace is not None
                and self.trace.has_user_message
                and not self.trace.error
                else None
            )
            reason = "interrupted" if self.agent.abort.is_set() else "finished"
            self.agent.finish(reason, trace_path=trace_path)
            if self.trace is not None:
                self.trace.discard_if_empty()
            self.agent.close()
            self._closed_session = True
        except Exception as exc:  # noqa: BLE001  # shutdown should always return the terminal
            error = f"{type(exc).__name__}: {exc}"
        self.post_message(SessionClosed(error))

    @on(SessionClosed)
    def on_session_closed(self, message: SessionClosed) -> None:
        self.exit(
            result=0 if message.error is None else 1,
            return_code=0 if message.error is None else 1,
            message=Text(message.error, style="red") if message.error else None,
        )

    def emergency_close(self) -> None:
        """Best-effort cleanup if the screen exits outside the normal command path."""
        if self._closed_session:
            return
        self.agent.interrupt()
        self.policy.prompter = None
        with contextlib.suppress(Exception):
            self.agent.finish("incomplete")
        if self.trace is not None:
            self.trace.discard_if_empty()
        with contextlib.suppress(Exception):
            self.agent.close()
        self._closed_session = True

    def _session_panel(self, event: RunStarted) -> Panel:
        """Build a compact, scannable startup status block."""
        status = Table.grid(padding=(0, 1), expand=False)
        status.add_column(style="bold cyan", no_wrap=True)
        status.add_column()
        status.add_column(style="bold cyan", no_wrap=True)
        status.add_column()
        status.add_row(
            "MODEL",
            event.model,
            "CONTEXT",
            f"{self.config.context_window:,} tokens",
        )
        status.add_row(
            "APPROVAL",
            self.config.approval_mode,
            "SANDBOX",
            self.agent.sandbox_status(),
        )
        status.add_row("PATHS", self._path_boundary_status(), "", "")
        status.add_row("SHELL", self._shell_execution_status(), "", "")
        status.add_row("ENDPOINT", event.endpoint, "", "")
        status.add_row("WORKSPACE", str(self.config.workspace), "", "")
        status.add_row(
            "TOOLS",
            f"{len(event.tool_names)} available · {', '.join(event.tool_names)}",
            "",
            "",
        )
        return Panel(
            status,
            title=Text("CAGENT  /  SESSION READY", style="bold cyan"),
            title_align="left",
            border_style="cyan",
            box=box.SQUARE,
            padding=(0, 1),
            expand=False,
        )

    def _path_boundary_status(self) -> str:
        """Describe whether file tools enforce the workspace boundary."""
        if self.config.allow_outside_workspace and self.agent.sandbox is None:
            return "unrestricted"
        return "workspace-only"

    def _shell_execution_status(self) -> str:
        """Describe the process boundary used by ``run_bash``."""
        if self.agent.sandbox is not None:
            return "container"
        return "host (unrestricted)"

    def _write(self, renderable: Any) -> None:
        """Append a Rich renderable as plain selectable transcript text."""
        transcript = self.query_one("#conversation", TextArea)
        text = self._renderable_text(renderable)
        if not text:
            return
        separator = "\n" if transcript.text else ""
        transcript.insert(separator + text, transcript.document.end)
        if self.focused is not transcript or not transcript.selected_text:
            transcript.scroll_end(animate=False)

    def _renderable_text(self, renderable: Any) -> str:
        """Render without ANSI escapes so terminal selection copies clean text."""
        if isinstance(renderable, str):
            return renderable.rstrip("\n")
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=max(min(self.size.width - 4, 120), 40),
            color_system=None,
            force_terminal=False,
            highlight=False,
            soft_wrap=False,
        )
        console.print(renderable)
        return buffer.getvalue().rstrip("\n")

    def _set_activity(self, text: str) -> None:
        self.query_one("#activity", Static).update(text)

    def _set_busy(self, busy: bool, activity: str) -> None:
        self._busy = busy
        self._set_activity(activity)
        composer = self.query_one("#composer", ComposerInput)
        composer.set_prompt_hint(
            "Agent is working · Ctrl+C interrupts" if busy else "Ask cagent or enter /help"
        )

    def _update_status(self) -> None:
        tokens = self.agent.context.token_count()
        sandbox = self.agent.sandbox_status()
        restored = f" · resumed {self._restored_from}" if self._restored_from else ""
        self.query_one("#activity", Static).update(
            f"{'Working' if self._busy else 'Ready'} · {self.config.approval_mode} · "
            f"sandbox {sandbox} · shell {self._shell_execution_status()} · "
            f"paths {self._path_boundary_status()} · "
            f"context {tokens:,}/{self.config.context_window:,}{restored}"
        )


def run_tui_session(
    config: AgentConfig,
    *,
    quiet: bool = False,
    show_thinking: bool = True,
) -> int:
    """Run one full-screen interactive session and always release its resources."""
    app = CagentTui(config, quiet=quiet, show_thinking=show_thinking)
    try:
        result = app.run()
        return result if isinstance(result, int) else 0
    finally:
        app.emergency_close()
