"""Recording a run so it can be inspected afterwards.

An agent's behaviour is hard to debug from its terminal output, which is written
for reading and discards the numbers. The trace is the other half: one JSON
object per event, appended as the run proceeds, so a session that ended badly can
be examined without reproducing it — which for a non-deterministic system is
often impossible.

Written as JSON Lines, flushed per event, because the runs worth examining are
usually the ones that crashed or were interrupted. A trailing partial line is
tolerable; a buffered file that lost the last minute is not.

Sensitive values never enter the trace: the recorded config omits the API key,
and tool metadata is serialised structurally rather than by ``repr``.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..types import Message, TextPart, ThinkingPart, ToolCallPart, ToolResultPart
from .events import (
    AgentEvent,
    ApprovalDecided,
    ApprovalRequested,
    CompactionDone,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    UserMessage,
    Warning,
)

__all__ = ["TraceWriter", "history_from_trace", "read_trace"]

_TRACE_VERSION = 1
_MAX_FIELD_CHARS = 20_000
"""Ceiling per recorded string, so one huge tool result cannot make a trace
unreadable. Truncation is marked so the reader knows it happened."""


def _clip(text: str) -> str:
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    dropped = len(text) - _MAX_FIELD_CHARS
    return text[:_MAX_FIELD_CHARS] + f"…[{dropped} chars omitted from trace]"


def _encode_message(message: Message) -> dict[str, Any]:
    """Render a message structurally, part by part."""
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": _clip(part.text)})
        elif isinstance(part, ThinkingPart):
            parts.append(
                {
                    "type": "thinking",
                    "text": _clip(part.text),
                    # Provider signatures are opaque verification material;
                    # clipping one would make an otherwise resumable turn
                    # invalid, so retain it verbatim.
                    "signature": part.signature,
                }
            )
        elif isinstance(part, ToolCallPart):
            parts.append(
                {
                    "type": "tool_call",
                    "id": part.id,
                    "name": part.name,
                    "arguments": _jsonable(part.arguments),
                    "raw_arguments": _clip(part.raw_arguments) if part.raw_arguments else None,
                }
            )
        elif isinstance(part, ToolResultPart):
            parts.append(
                {
                    "type": "tool_result",
                    "call_id": part.call_id,
                    "content": _clip(part.content),
                    "is_error": part.is_error,
                }
            )
    encoded: dict[str, Any] = {"role": message.role, "parts": parts}
    if message.synthetic:
        encoded["synthetic"] = True
    return encoded


def _jsonable(value: object) -> Any:
    """Coerce arbitrary tool metadata into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return _clip(str(value))


@dataclass(slots=True)
class TraceWriter:
    """Appends run events to a JSONL file.

    A trace is best-effort infrastructure: if the file cannot be written the run
    continues without it, recording the reason in :attr:`error`. Losing
    observability is not worth losing the user's task.
    """

    path: Path
    started: float = field(default_factory=time.time)
    events_written: int = 0
    error: str | None = None
    has_user_message: bool = False
    config: AgentConfig | None = field(default=None, repr=False)
    _pending: list[AgentEvent] = field(default_factory=list, repr=False)
    _pending_history: list[Message] | None = field(default=None, repr=False)
    _handle: Any = None

    @classmethod
    def create(cls, config: AgentConfig, *, session_id: str) -> TraceWriter | None:
        """Prepare a lazy trace under the configured directory, or return ``None``.

        The file and parent directory are created only after the first real
        user turn. Returns ``None`` when tracing is switched off.
        """
        if config.trace_dir is None:
            return None
        path = config.trace_dir / f"{session_id}.jsonl"
        return cls(path=path, config=config)

    def _open(self, config: AgentConfig) -> None:
        if config is None:
            self.error = "trace writer has no configuration"
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            self.error = f"could not open {self.path}: {exc}"
            return
        self._write(
            {
                "type": "session",
                "version": _TRACE_VERSION,
                "model": config.model,
                "endpoint": config.base_url,
                "wire": config.wire,
                "reasoning_effort": config.reasoning_effort,
                "workspace": str(config.workspace),
                "approval_mode": config.approval_mode,
                "sandbox_mode": config.sandbox_mode,
                "sandbox_sync": config.sandbox_sync,
                "sandbox_network": config.sandbox_network,
                "context_window": config.context_window,
            }
        )

    def handle(self, event: AgentEvent) -> None:
        """Record one event. Never raises."""
        if isinstance(event, RunStarted):
            # Keep the startup metadata in memory until a real turn starts.
            self._pending.append(event)
            return
        if not self.has_user_message and not isinstance(event, UserMessage):
            return
        if isinstance(event, UserMessage):
            if not event.text.strip():
                return
            self.has_user_message = True
        self._ensure_open()
        record = self._encode(event)
        if record is not None:
            self._write(record)

    def record_history(self, messages: Sequence[Message]) -> None:
        """Record the history that future events continue from.

        An empty checkpoint is meaningful: commands such as ``/undo`` may
        remove the only user turn, and replay must not resurrect it from older
        append-only events.
        """
        self._pending_history = list(messages)
        if self._handle is None:
            return
        self._flush_pending_history()

    def _flush_pending_history(self) -> None:
        """Write a queued resume checkpoint once the trace is open."""
        messages, self._pending_history = self._pending_history, None
        if messages is None:
            return
        self._write(
            {
                "type": "history_checkpoint",
                "messages": [_encode_message(message) for message in messages],
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        record.setdefault("t", round(time.time() - self.started, 3))
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
            self.events_written += 1
        except (OSError, TypeError, ValueError) as exc:
            self.error = f"trace write failed: {exc}"
            self.close()

    def _ensure_open(self) -> None:
        """Open the file on the first event that represents real work."""
        if self._handle is not None or self.error is not None:
            return
        assert self.config is not None
        self._open(self.config)
        if self._handle is None:
            return
        pending, self._pending = self._pending, []
        for event in pending:
            record = self._encode(event)
            if record is not None:
                self._write(record)
        self._flush_pending_history()

    def discard_if_empty(self) -> None:
        """Remove the trace if no non-empty user turn was ever recorded.

        Empty sessions are startup metadata, not useful conversation history,
        and should not remain in the workspace.
        """
        path = self.path
        self.close()
        if self.has_user_message:
            return
        with contextlib.suppress(OSError):
            path.unlink()

    @staticmethod
    def _encode(event: AgentEvent) -> dict[str, Any] | None:
        """Map an event to a record, or ``None`` for events not worth storing.

        Text and thinking deltas are dropped: they are re-derivable from the
        assembled message in :class:`StepFinished`, and storing thousands of
        fragments would bury the events that carry information.
        """
        match event:
            case RunStarted():
                return {
                    "type": "run_started",
                    "task": _clip(event.task),
                    "model": event.model,
                    "endpoint": event.endpoint,
                    "system_tokens": event.system_tokens,
                    "tools": list(event.tool_names),
                    "sandbox_status": event.sandbox_status,
                    "shell_access": event.shell_access,
                    "path_boundary": event.path_boundary,
                }
            case UserMessage():
                return {"type": "user", "text": _clip(event.text)}
            case StepStarted():
                return {
                    "type": "step_started",
                    "step": event.step,
                    "prompt_tokens_estimate": event.prompt_tokens_estimate,
                }
            case StepFinished():
                return {
                    "type": "step_finished",
                    "step": event.step,
                    "finish_reason": event.finish_reason,
                    "latency_s": round(event.latency_s, 3),
                    "usage": {
                        "prompt": event.usage.prompt_tokens,
                        "completion": event.usage.completion_tokens,
                        "cached": event.usage.cached_tokens,
                        "reasoning": event.usage.reasoning_tokens,
                    },
                    "message": _encode_message(event.message),
                }
            case ApprovalRequested():
                return {
                    "type": "approval_requested",
                    "tool": event.request.tool,
                    "risk": event.request.risk.name,
                    "summary": _clip(event.request.summary),
                    "signature": event.request.signature,
                    "always_prompt": event.request.always_prompt,
                }
            case ApprovalDecided():
                return {
                    "type": "approval_decided",
                    "tool": event.request.tool,
                    "approved": event.approved,
                    "remembered": event.remembered,
                    "automatic": event.automatic,
                }
            case ToolStarted():
                return {
                    "type": "tool_started",
                    "id": event.call.id,
                    "name": event.call.name,
                    "risk": event.risk.name,
                    "arguments": _jsonable(event.call.arguments),
                }
            case ToolFinished():
                return {
                    "type": "tool_finished",
                    "id": event.call.id,
                    "name": event.call.name,
                    "is_error": event.outcome.is_error,
                    "truncated": event.outcome.truncated,
                    "duration_s": round(event.duration_s, 3),
                    "content": _clip(event.outcome.content),
                    "metadata": _jsonable(event.outcome.metadata),
                }
            case CompactionDone():
                return {
                    "type": "compaction",
                    "strategy": event.strategy,
                    "tokens_before": event.tokens_before,
                    "tokens_after": event.tokens_after,
                    "messages_before": event.messages_before,
                    "messages_after": event.messages_after,
                }
            case Warning():
                return {
                    "type": "warning",
                    "message": _clip(event.message),
                    "detail": _clip(event.detail) if event.detail else None,
                }
            case TurnFinished():
                return {
                    "type": "turn_finished",
                    "steps": event.steps,
                    "reply": _clip(event.reply),
                    "usage": {
                        "prompt": event.usage.prompt_tokens,
                        "completion": event.usage.completion_tokens,
                    },
                }
            case RunFinished():
                return {
                    "type": "run_finished",
                    "reason": event.reason,
                    "steps": event.steps,
                    "elapsed_s": round(event.elapsed_s, 3),
                    "usage": {
                        "prompt": event.usage.prompt_tokens,
                        "completion": event.usage.completion_tokens,
                        "cached": event.usage.cached_tokens,
                        "reasoning": event.usage.reasoning_tokens,
                    },
                }
            case _:
                return None

    def close(self) -> None:
        """Close the file. Idempotent."""
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a trace, skipping any unparseable trailing line.

    A truncated final record is expected — it means the process died mid-write,
    which is exactly the case a trace exists to explain.
    """
    records: list[dict[str, Any]] = []
    # ``utf-8-sig`` keeps normal UTF-8 behaviour and also accepts traces saved
    # by Windows editors that prepend a BOM.
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _decode_message(raw: object) -> Message | None:
    """Decode one trace message, tolerating fields from older trace versions."""
    if not isinstance(raw, Mapping):
        return None
    role = raw.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None
    raw_parts = raw.get("parts")
    if not isinstance(raw_parts, list):
        return None
    parts: list[TextPart | ThinkingPart | ToolCallPart | ToolResultPart] = []
    for item in raw_parts:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            parts.append(TextPart(item["text"]))
        elif kind == "thinking" and isinstance(item.get("text"), str):
            signature = item.get("signature")
            parts.append(
                ThinkingPart(
                    item["text"], signature if isinstance(signature, str) else None
                )
            )
        elif kind == "tool_call":
            call_id = item.get("id")
            name = item.get("name")
            arguments = item.get("arguments")
            raw_arguments = item.get("raw_arguments")
            if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, dict):
                parts.append(
                    ToolCallPart(
                        call_id,
                        name,
                        dict(arguments),
                        raw_arguments if isinstance(raw_arguments, str) else "",
                    )
                )
        elif kind == "tool_result":
            call_id = item.get("call_id")
            content = item.get("content")
            if isinstance(call_id, str) and isinstance(content, str):
                parts.append(
                    ToolResultPart(call_id, content, bool(item.get("is_error", False)))
                )
    return (
        Message(role=role, parts=parts, synthetic=bool(raw.get("synthetic", False)))
        if parts
        else None
    )


def history_from_trace(records: Sequence[Mapping[str, Any]]) -> list[Message]:
    """Reconstruct a provider-valid conversation history from trace events.

    A trace stores assistant messages in ``step_finished`` records and tool
    outcomes in ``tool_finished`` records.  The latter are grouped back into a
    single tool message, matching the history shape sent to both wire adapters.
    An interrupted final tool call is dropped instead of producing an invalid
    assistant message with no results.
    """
    history: list[Message] = []
    pending: list[ToolResultPart] = []

    def flush_tools() -> None:
        if not history or history[-1].role != "assistant" or not history[-1].tool_calls:
            pending.clear()
            return
        if not pending:
            # A user/assistant event after this call means the process ended
            # before any result was recorded. Keep the transcript provider-safe.
            history.pop()
            return
        by_id = {result.call_id: result for result in pending}
        calls = history[-1].tool_calls
        expected = {call.id for call in calls}
        if expected <= by_id.keys():
            history.append(Message.from_tool_results([by_id[call.id] for call in calls]))
        else:
            # The process may have died between the model response and tool
            # execution. Do not hand an incomplete tool turn to the provider.
            history.pop()
        pending.clear()

    for record in records:
        kind = record.get("type")
        if kind == "history_checkpoint":
            raw_messages = record.get("messages")
            if isinstance(raw_messages, list):
                restored = [
                    message
                    for raw_message in raw_messages
                    if (message := _decode_message(raw_message)) is not None
                ]
                # An explicitly empty checkpoint clears history. If a non-empty
                # checkpoint is entirely malformed, retain the earlier valid
                # history instead of treating corruption as an undo.
                if restored or not raw_messages:
                    history = restored
                    pending.clear()
        elif kind == "user":
            flush_tools()
            text = record.get("text")
            if isinstance(text, str):
                history.append(Message.user(text))
        elif kind == "step_finished":
            flush_tools()
            message = _decode_message(record.get("message"))
            if message is not None:
                history.append(message)
        elif kind == "tool_finished":
            call_id = record.get("id")
            content = record.get("content")
            if isinstance(call_id, str) and isinstance(content, str):
                pending.append(
                    ToolResultPart(call_id, content, bool(record.get("is_error", False)))
                )
    flush_tools()
    if history and history[-1].role == "assistant" and history[-1].tool_calls:
        history.pop()
    return history
