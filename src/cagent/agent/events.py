"""What the engine reports as it works.

The engine never prints. It emits typed events to an :class:`EventSink`, and
something else decides what a human should see. That split is what lets the same
run drive a terminal UI, a JSONL trace file, and a test that asserts on a list —
concurrently, because a sink can fan out to several others.

The events are a closed union, so a sink can match on them exhaustively and a
new event type cannot be silently ignored by an existing renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

from ..tools.base import ApprovalRequest, ToolOutcome
from ..types import Message, RiskLevel, ToolCallPart, Usage

__all__ = [
    "AgentEvent",
    "ApprovalDecided",
    "ApprovalRequested",
    "CollectingSink",
    "CompactionDone",
    "EventSink",
    "FanOutSink",
    "RunFinished",
    "RunStarted",
    "StepFinished",
    "StepStarted",
    "TextDelta",
    "ThinkingDelta",
    "ToolFinished",
    "ToolStarted",
    "TurnFinished",
    "UserMessage",
    "Warning",
]


_E = TypeVar("_E")


@dataclass(frozen=True, slots=True)
class RunStarted:
    """A task was accepted. ``system_tokens`` includes the repo map."""

    task: str
    model: str
    endpoint: str
    """The base URL requests go to — the fact worth recording, since the same
    model name can be served by several endpoints."""

    system_tokens: int
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UserMessage:
    """The user's turn was appended to the transcript."""

    text: str


@dataclass(frozen=True, slots=True)
class StepStarted:
    """One iteration of the loop begins: a request is about to be sent."""

    step: int
    max_steps: int
    prompt_tokens_estimate: int


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A fragment of the model's reasoning trace."""

    text: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of the model's prose answer."""

    text: str


@dataclass(frozen=True, slots=True)
class StepFinished:
    """The model's response for this step is complete."""

    step: int
    message: Message
    finish_reason: str
    usage: Usage
    latency_s: float


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    """The user is being asked to permit a call."""

    request: ApprovalRequest


@dataclass(frozen=True, slots=True)
class ApprovalDecided:
    """How an approval resolved, and whether the answer was remembered."""

    request: ApprovalRequest
    approved: bool
    remembered: bool = False
    automatic: bool = False
    """True when the policy answered without consulting the user."""


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A tool call is about to execute."""

    call: ToolCallPart
    risk: RiskLevel


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool call finished, successfully or not."""

    call: ToolCallPart
    outcome: ToolOutcome
    duration_s: float


@dataclass(frozen=True, slots=True)
class CompactionDone:
    """History was compacted to fit the context window."""

    strategy: str
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int


@dataclass(frozen=True, slots=True)
class Warning:
    """Something recoverable happened that the user should know about."""

    message: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class TurnFinished:
    """The agent finished replying to one user turn."""

    reply: str
    steps: int
    usage: Usage


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The session ended. ``reason`` explains why the loop stopped."""

    reason: str
    steps: int
    usage: Usage
    elapsed_s: float
    trace_path: str | None = None


AgentEvent = (
    RunStarted
    | UserMessage
    | StepStarted
    | ThinkingDelta
    | TextDelta
    | StepFinished
    | ApprovalRequested
    | ApprovalDecided
    | ToolStarted
    | ToolFinished
    | CompactionDone
    | Warning
    | TurnFinished
    | RunFinished
)
"""Everything the engine can report. Exhaustive: sinks may match on it."""


@runtime_checkable
class EventSink(Protocol):
    """Anything that can consume engine events."""

    def handle(self, event: AgentEvent) -> None:
        """Consume one event. Must not raise: the engine keeps working."""


@dataclass(slots=True)
class CollectingSink:
    """Records every event, for tests and for end-of-run summaries."""

    events: list[AgentEvent] = field(default_factory=list)

    def handle(self, event: AgentEvent) -> None:
        self.events.append(event)

    def of_type(self, kind: type[_E]) -> list[_E]:
        """Every recorded event of one type, in order."""
        return [event for event in self.events if isinstance(event, kind)]


@dataclass(slots=True)
class FanOutSink:
    """Forwards events to several sinks.

    A sink that raises is dropped rather than allowed to abort the run: a broken
    renderer or an unwritable trace file must not lose the user's work.
    """

    sinks: list[EventSink] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def handle(self, event: AgentEvent) -> None:
        for sink in list(self.sinks):
            try:
                sink.handle(event)
            except Exception as exc:  # noqa: BLE001  # a sink must never stop the run
                self.sinks.remove(sink)
                self.failures.append(f"{type(sink).__name__}: {type(exc).__name__}: {exc}")
