"""Newline-delimited JSON bridge used by the Electron desktop client.

The bridge deliberately stays thin: the Python agent remains the owner of
context, approvals, tools, sandboxing, and traces. Electron only transports
commands and renders the typed events emitted here.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import enum
import io
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

from ..agent.approval import ApprovalPolicy, Decision
from ..agent.engine import Agent, TurnResult
from ..agent.events import AgentEvent, EventSink, FanOutSink
from ..agent.trace import TraceWriter, history_from_trace, read_trace
from ..cli.app import _command
from ..cli.resume import (
    find_trace_choices,
    first_user_prompt,
    resume_trace_dir,
    trace_step_count,
)
from ..config import AgentConfig, load_config
from ..tools.base import ApprovalRequest
from ..tools.registry import default_registry


def _snake(name: str) -> str:
    result: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _jsonable(value: object) -> Any:
    if isinstance(value, enum.Enum):
        return value.name.lower()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded = {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        part_names = {
            "TextPart": "text",
            "ThinkingPart": "thinking",
            "ToolCallPart": "tool_call",
            "ToolResultPart": "tool_result",
        }
        part_type = part_names.get(type(value).__name__)
        if part_type is not None:
            return {"type": part_type, **encoded}
        return encoded
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class JsonEmitter:
    """Serialise output from agent and command worker threads atomically."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def send(self, kind: str, **payload: object) -> None:
        record = {"type": kind, **{key: _jsonable(value) for key, value in payload.items()}}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


@dataclasses.dataclass(slots=True)
class BridgeEventSink(EventSink):
    emitter: JsonEmitter

    def handle(self, event: AgentEvent) -> None:
        self.emitter.send(_snake(type(event).__name__), **_jsonable(event))


class DesktopSession:
    def __init__(self, workspace: Path, emitter: JsonEmitter) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.emitter = emitter
        self.config: AgentConfig | None = None
        self.agent: Agent | None = None
        self.policy: ApprovalPolicy | None = None
        self.trace: TraceWriter | None = None
        self.sink: FanOutSink | None = None
        self._worker: threading.Thread | None = None
        self._busy_lock = threading.Lock()
        self._busy = False
        self._initialized = False
        self._closed = False
        self._shutdown_started = False
        self._approval_lock = threading.Lock()
        self._approval_ready = threading.Event()
        self._approval_request: ApprovalRequest | None = None
        self._approval_decision: Decision | None = None
        self._restored_from: str | None = None
        self._restored_steps = 0

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._launch(self._initialize, name="cagent-gui-initialize", allow_uninitialized=True)

    def _initialize(self) -> None:
        self.emitter.send("activity", message="Building repository context")
        try:
            config = load_config(cwd=self.workspace)
            if config.trace_dir is None:
                config.trace_dir = config.workspace / ".cagent" / "traces"
            config.validate()
            policy = ApprovalPolicy(config, prompter=self._prompt_for_approval)
            event_sink = BridgeEventSink(self.emitter)
            sink = FanOutSink([event_sink])
            agent = Agent.create(
                config,
                sink=sink,
                policy=policy,
                registry=default_registry(),
                defer_initial_prompt=True,
            )
            trace = TraceWriter.create(config, session_id=agent.session_id)
            if trace is not None:
                sink.sinks.append(trace)
            agent.initialize()
            self.config = config
            self.agent = agent
            self.policy = policy
            self.trace = trace
            self.sink = sink
            self._initialized = True
            agent.announce("(desktop)")
            self.emitter.send(
                "ready",
                session_id=agent.session_id,
                workspace=str(config.workspace),
                config=self._config_summary(),
            )
            self.send_sessions()
            self.send_status()
        except Exception as exc:  # noqa: BLE001 - startup errors belong in the GUI
            self.emitter.send(
                "fatal_error",
                message=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(limit=8),
            )

    def close(self, reason: str = "finished", *, interactive: bool = False) -> None:
        """Finish the session, applying the sandbox sync policy on the way out.

        Args:
            reason: Recorded in the trace as why the session ended.
            interactive: Whether a renderer is still there to answer questions.
                Closing a sandboxed session asks whether to copy the disposable
                copy back to the project; with no one to ask, the policy refuses
                and the work is thrown away. That is the right answer for a
                renderer that is already gone, and the wrong one for a session
                being rotated under a live window — so the caller decides.
        """
        if self._closed:
            return
        self._closed = True
        self.interrupt()
        agent = self.agent
        if agent is None:
            return
        if self.policy is not None and not interactive:
            self.policy.prompter = None
        trace_path = (
            str(self.trace.path)
            if self.trace is not None and self.trace.has_user_message and not self.trace.error
            else None
        )
        with contextlib.suppress(Exception):
            agent.finish(reason, trace_path=trace_path)
        if self.trace is not None:
            self.trace.discard_if_empty()
        with contextlib.suppress(Exception):
            agent.close()

    def begin_shutdown(self) -> None:
        """Close the session off the reader thread, then report completion.

        Closing inline would deadlock: the sandbox sync approval is answered by
        an ``approval`` message over stdin, and the reader thread cannot deliver
        it while it is itself blocked inside :meth:`close`. Electron waits for
        ``shutdown_complete`` before it tears the window down.
        """
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.interrupt()

        def run() -> None:
            try:
                self.close("shutdown", interactive=True)
            except Exception as exc:  # noqa: BLE001 - shutdown must always finish
                self.emitter.send(
                    "worker_error",
                    message=f"{type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(limit=8),
                )
            finally:
                self.emitter.send("shutdown_complete")

        threading.Thread(target=run, name="cagent-gui-shutdown", daemon=True).start()

    def new_session(self) -> None:
        if not self._require_idle():
            return
        self._launch(self._new_session, name="cagent-gui-new-session")

    def _new_session(self) -> None:
        self.emitter.send("history_cleared")
        self._rotate_session("new_session")

    def _rotate_session(self, reason: str) -> None:
        """Finish the live session and build a fresh one in this same process."""
        # The window is open and the user is waiting on us, so the sandbox sync
        # question can and must be asked: rotating a session is not a reason to
        # silently discard everything the agent wrote.
        self.close(reason, interactive=True)
        # Re-use this bridge process and the resolved workspace, but create all
        # mutable agent collaborators again so usage and remembered approvals reset.
        self._closed = False
        self._initialized = False
        self.agent = None
        self.policy = None
        self.trace = None
        self.sink = None
        self._restored_from = None
        self._restored_steps = 0
        self._initialize()

    # ------------------------------------------------------------------ turns

    def run_turn(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            self.run_command(text)
            return
        if not self._require_idle():
            return
        self._launch(lambda: self._run_turn(text), name="cagent-gui-turn")

    def _run_turn(self, text: str) -> None:
        assert self.agent is not None
        self.agent.reset_interrupt()
        result: TurnResult = self.agent.run_turn(text)
        self.emitter.send(
            "turn_complete",
            stopped_by=result.stopped_by,
            completed=result.completed,
        )
        self.send_status()
        self.send_sessions()

    def interrupt(self) -> None:
        if self.agent is not None:
            self.agent.interrupt()
        self.resolve_approval("abort")
        self.emitter.send("activity", message="Interrupting current work")

    # --------------------------------------------------------------- approvals

    def _prompt_for_approval(self, request: ApprovalRequest) -> Decision:
        with self._approval_lock:
            self._approval_request = request
            self._approval_decision = None
            self._approval_ready.clear()
        self._approval_ready.wait()
        with self._approval_lock:
            decision = self._approval_decision or Decision(False, abort=True)
            self._approval_request = None
            self._approval_decision = None
        return decision

    def resolve_approval(self, answer: str) -> None:
        decisions = {
            "approve": Decision(True),
            "deny": Decision(False),
            "always": Decision(True, remember=True),
            "abort": Decision(False, abort=True),
        }
        decision = decisions.get(answer)
        if decision is None:
            self.emitter.send("protocol_error", message=f"Unknown approval decision: {answer}")
            return
        with self._approval_lock:
            if self._approval_request is None:
                return
            if (
                answer == "always"
                and (
                    self._approval_request.risk.name == "DANGEROUS"
                    or self._approval_request.always_prompt
                )
            ):
                self.emitter.send(
                    "protocol_error", message="This approval cannot be remembered."
                )
                return
            self._approval_decision = decision
            self._approval_ready.set()

    # -------------------------------------------------------------- slash cmds

    def run_command(self, line: str) -> None:
        if not self._require_idle():
            return
        name = line.split(maxsplit=1)[0].lower()
        if name == "/resume":
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                self.send_sessions()
            else:
                self.resume(Path(parts[1].strip().strip('"')))
            return
        if name in ("/exit", "/quit"):
            self.emitter.send("exit_requested")
            return
        self._launch(lambda: self._run_command(line), name="cagent-gui-command")

    def _run_command(self, line: str) -> None:
        assert self.agent is not None and self.config is not None
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=110,
            highlight=False,
            soft_wrap=False,
            color_system=None,
        )
        try:
            _command(console, self.agent, self.config, line)
            output, error = buffer.getvalue().rstrip(), None
        except Exception as exc:  # noqa: BLE001 - commands must leave the GUI usable
            output = buffer.getvalue().rstrip()
            error = f"{type(exc).__name__}: {exc}"
        if line.split(maxsplit=1)[0].lower() == "/clear":
            if self.trace is not None:
                self.trace.record_history(self.agent.context.history)
            self.emitter.send("history_cleared")
        elif line.split(maxsplit=1)[0].lower() == "/undo" and error is None:
            self.emitter.send(
                "history_restored",
                messages=self.agent.context.history,
                source="undo",
                warning=(
                    "The latest user turn was removed from model context. "
                    "File and command effects were not reverted."
                ),
            )
        self.emitter.send("command_result", command=line, output=output, error=error)
        self.send_status()

    # ---------------------------------------------------------------- sessions

    def send_sessions(self) -> None:
        if self.config is None:
            return
        choices = find_trace_choices(resume_trace_dir(self.config))
        sessions = [
            {
                "id": choice.session_id,
                "path": str(choice.path),
                "modified": dt.datetime.fromtimestamp(choice.modified, tz=dt.UTC)
                .astimezone()
                .isoformat(),
                "prompt": " ".join(choice.prompt.split()),
                "steps": choice.steps,
                "status": choice.status,
            }
            for choice in choices
        ]
        self.emitter.send(
            "sessions",
            sessions=sessions,
            active_id=self.agent.session_id if self.agent is not None else None,
            restored_from=self._restored_from,
        )

    def delete_session(self, path: Path) -> None:
        """Remove a saved trace so it stops appearing in the session list.

        The renderer supplies the path, so the deletion is fenced twice. Only
        ``*.jsonl`` files directly inside this workspace's trace directory can
        go — a session list is not a reason to hand the GUI an arbitrary unlink.
        The live session's own trace is refused as well: its writer still holds
        the file open, and removing it mid-run would discard the record of the
        turn currently in flight rather than an old conversation.

        Runs on the reader thread rather than through :meth:`_launch`: it is one
        filesystem call, and tidying the sidebar should not have to wait for the
        agent to finish thinking.
        """
        if self.config is None:
            self.emitter.send("protocol_error", message="The agent is still initializing.")
            return
        trace_dir = resume_trace_dir(self.config)
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = trace_dir / candidate
        try:
            candidate = candidate.resolve()
        except OSError as exc:
            self.emitter.send("protocol_error", message=f"Could not resolve {candidate}: {exc}")
            return
        if candidate.suffix != ".jsonl" or candidate.parent != trace_dir:
            self.emitter.send(
                "protocol_error",
                message=f"Only traces stored in {trace_dir} can be deleted.",
            )
            return
        live = self.trace.path.expanduser().resolve() if self.trace is not None else None
        if live is not None and candidate == live:
            self.emitter.send(
                "protocol_error",
                message="That is the session you are in — start a new session first.",
            )
            return
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass  # Already gone; the refreshed list below is the answer either way.
        except OSError as exc:
            self.emitter.send(
                "protocol_error",
                message=f"Could not delete {candidate.name}: {exc}",
            )
            return
        self.emitter.send("session_deleted", path=str(candidate), session=candidate.stem)
        self.send_sessions()

    def resume(self, path: Path) -> None:
        if not self._require_idle() or self.config is None or self.agent is None:
            return
        self._launch(lambda: self._resume(path), name="cagent-gui-resume")

    def _resume(self, path: Path) -> None:
        assert self.config is not None
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = resume_trace_dir(self.config) / candidate
        try:
            candidate = candidate.resolve()
            records = read_trace(candidate)
        except OSError as exc:
            self.emitter.send("resume_error", message=f"Could not read {candidate}: {exc}")
            return
        if first_user_prompt(records) is None:
            self.emitter.send("resume_error", message="This trace has no user request.")
            return
        history = history_from_trace(records)
        if not history:
            self.emitter.send("resume_error", message="This trace has no restorable history.")
            return
        recorded_workspace = next(
            (record.get("workspace") for record in records if record.get("type") == "session"),
            None,
        )
        warning = None
        if (
            isinstance(recorded_workspace, str)
            and Path(recorded_workspace).expanduser().resolve() != self.config.workspace
        ):
            warning = (
                f"This trace used {recorded_workspace}; tools will continue to use "
                f"{self.config.workspace}."
            )
        # `restore_history` replaces the whole transcript and documents that a
        # resumed context belongs to a fresh agent. The desktop client can ask
        # for a resume mid-conversation, so rotate first: grafting the history
        # onto a session that already has turns writes a history_checkpoint
        # into that session's own trace, and the checkpoint then wins on replay
        # and hides the conversation the trace was recording.
        if self.trace is not None and self.trace.has_user_message:
            self._rotate_session("resumed")
        if self.agent is None:
            return
        self.agent.restore_history(history)
        if self.trace is not None:
            self.trace.record_history(history)
        self._restored_from = candidate.stem
        # The rotated-in agent starts with a fresh LoopGuard, which is right for
        # the step *budget* but wrong for the step *count* the status bar shows:
        # the conversation on screen did not begin at zero. Carry the trace's own
        # count so the footer agrees with the sidebar card it was restored from.
        self._restored_steps = trace_step_count(records)
        self.emitter.send(
            "history_restored",
            messages=history,
            source=candidate.stem,
            warning=warning,
        )
        self.send_status()
        self.send_sessions()

    # ----------------------------------------------------------------- status

    def send_status(self) -> None:
        if self.agent is None or self.config is None:
            return
        self.emitter.send(
            "status",
            busy=self._busy,
            session_id=self.agent.session_id,
            restored_from=self._restored_from,
            usage=self.agent.usage,
            # Steps in this conversation, not in this agent: see `_restored_steps`.
            steps=self._restored_steps + self.agent.guard.steps,
            context={
                "tokens": self.agent.context.token_count(),
                "window": self.config.context_window,
                "messages": len(self.agent.context.history),
                "compactions": self.agent.context.compactions,
            },
            config=self._config_summary(),
        )

    def _config_summary(self) -> dict[str, object]:
        assert self.config is not None and self.agent is not None
        return {
            "model": self.config.model or "not configured",
            "endpoint": self.config.base_url or "not configured",
            "wire": self.config.wire,
            "workspace": str(self.config.workspace),
            "approval_mode": self.config.approval_mode,
            "reasoning_effort": self.config.reasoning_effort or "default",
            "sandbox": self.agent.sandbox_status(),
            "sandbox_mode": self.config.sandbox_mode,
            "sandbox_sync": self.config.sandbox_sync,
            "sandbox_image": self.config.sandbox_image,
            "sandbox_network": self.config.sandbox_network,
            "trace_dir": str(self.config.trace_dir) if self.config.trace_dir else None,
            "context_window": self.config.context_window,
            "tools": sorted(self.agent.registry.names()),
        }

    # ---------------------------------------------------------------- workers

    def _require_idle(self) -> bool:
        if not self._initialized:
            self.emitter.send("protocol_error", message="The agent is still initializing.")
            return False
        with self._busy_lock:
            if self._busy:
                self.emitter.send(
                    "protocol_error",
                    message="Finish or interrupt the current operation first.",
                )
                return False
        return True

    def _launch(
        self,
        target: Any,
        *,
        name: str,
        allow_uninitialized: bool = False,
    ) -> None:
        with self._busy_lock:
            if self._busy:
                self.emitter.send("protocol_error", message="The agent is already working.")
                return
            if not allow_uninitialized and not self._initialized:
                self.emitter.send("protocol_error", message="The agent is still initializing.")
                return
            self._busy = True
        self.emitter.send("busy_changed", busy=True)

        def run() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - retain the bridge after worker failure
                self.emitter.send(
                    "worker_error",
                    message=f"{type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(limit=8),
                )
            finally:
                with self._busy_lock:
                    self._busy = False
                self.emitter.send("busy_changed", busy=False)
                if self._initialized:
                    self.send_status()

        self._worker = threading.Thread(target=run, name=name, daemon=True)
        self._worker.start()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cagent Electron JSONL bridge")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser


def _force_utf8_streams() -> None:
    """Pin the JSONL pipes to UTF-8.

    When stdio is a pipe Python falls back to the locale encoding (cp936 on a
    Chinese Windows install), while Electron always reads and writes UTF-8.
    Anything non-ASCII would be mangled in both directions.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    args = _parser().parse_args(argv)
    emitter = JsonEmitter()
    session = DesktopSession(args.workspace, emitter)
    session.start()
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("command must be a JSON object")
                kind = message.get("type")
                if kind == "turn":
                    session.run_turn(str(message.get("text", "")))
                elif kind == "command":
                    session.run_command(str(message.get("command", "")))
                elif kind == "interrupt":
                    session.interrupt()
                elif kind == "approval":
                    session.resolve_approval(str(message.get("decision", "")))
                elif kind == "sessions":
                    session.send_sessions()
                elif kind == "resume":
                    session.resume(Path(str(message.get("path", ""))))
                elif kind == "delete_session":
                    session.delete_session(Path(str(message.get("path", ""))))
                elif kind == "new_session":
                    session.new_session()
                elif kind == "status":
                    session.send_status()
                elif kind == "shutdown":
                    # Keep reading: the sandbox sync approval still has to come
                    # back over this stream. The loop ends on stdin EOF, which
                    # Electron sends once it sees `shutdown_complete`.
                    session.begin_shutdown()
                else:
                    emitter.send("protocol_error", message=f"Unknown command type: {kind!r}")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                emitter.send("protocol_error", message=str(exc))
    finally:
        session.close("bridge_closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
