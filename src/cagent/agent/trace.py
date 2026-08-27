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

__all__ = ["TraceWriter", "read_trace"]

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
            parts.append({"type": "thinking", "text": _clip(part.text)})
        elif isinstance(part, ToolCallPart):
            parts.append(
                {
                    "type": "tool_call",
                    "id": part.id,
                    "name": part.name,
                    "arguments": _jsonable(part.arguments),
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
    return {"role": message.role, "parts": parts}


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
    _handle: Any = None

    @classmethod
    def create(cls, config: AgentConfig, *, session_id: str) -> TraceWriter | None:
        """Open a trace under the configured directory, or return ``None``.

        Returns ``None`` when tracing is switched off, which lets the caller
        treat "no trace configured" and "trace unavailable" the same way.
        """
        if config.trace_dir is None:
            return None
        path = config.trace_dir / f"{session_id}.jsonl"
        writer = cls(path=path)
        writer._open(config)
        return writer

    def _open(self, config: AgentConfig) -> None:
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
                "provider": config.provider,
                "model": config.resolved_model,
                "wire": config.resolved_wire,
                "workspace": str(config.workspace),
                "approval_mode": config.approval_mode,
                "context_window": config.context_window,
                "max_steps": config.max_steps,
            }
        )

    def handle(self, event: AgentEvent) -> None:
        """Record one event. Never raises."""
        record = self._encode(event)
        if record is not None:
            self._write(record)

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
                    "provider": event.provider,
                    "system_tokens": event.system_tokens,
                    "tools": list(event.tool_names),
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
    with path.open(encoding="utf-8") as handle:
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
