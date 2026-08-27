"""Provider-neutral data model shared by every layer of the agent.

Only the standard library is imported here. Every other module depends on these
types, so keeping this one dependency-free avoids import cycles and lets the
wire adapters translate to and from vendor payloads without a common base class.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "ContentPart",
    "FinishReason",
    "Message",
    "RiskLevel",
    "Role",
    "TextPart",
    "ThinkingPart",
    "ToolCallPart",
    "ToolResultPart",
    "ToolSpec",
    "Usage",
]

Role = Literal["system", "user", "assistant", "tool"]
"""Who authored a message, in provider-neutral terms."""

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "aborted", "error"]
"""Why the model stopped generating; drives loop termination."""


@dataclass(frozen=True, slots=True)
class TextPart:
    """Prose emitted by, or addressed to, the model."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """One segment of a reasoning trace from a thinking-capable model.

    ``signature`` is an opaque provider blob. When present it must be echoed
    back verbatim on the next request or the provider rejects the transcript.
    """

    text: str
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, object]
    raw_arguments: str = ""
    """Unparsed argument JSON, retained so a repair prompt can quote it back."""


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The outcome of one tool invocation, keyed back to its originating call."""

    call_id: str
    content: str
    is_error: bool = False


ContentPart = TextPart | ThinkingPart | ToolCallPart | ToolResultPart
"""Any unit of message content. Exhaustive: adapters may match on it."""

@dataclass(slots=True)
class Message:
    """One turn of the transcript.

    Mutable by design: the context manager rewrites parts in place when it
    truncates tool output, and caches the token count it computed on the way.
    """

    role: Role
    parts: list[ContentPart] = field(default_factory=list)
    token_estimate: int | None = None
    """Cached token count, filled in by the context manager. ``None`` = stale."""

    @classmethod
    def user(cls, text: str) -> Message:
        """Build a user turn from plain text."""
        return cls(role="user", parts=[TextPart(text)])

    @classmethod
    def assistant(cls, *parts: ContentPart) -> Message:
        """Build an assistant turn from already-parsed content parts."""
        return cls(role="assistant", parts=list(parts))

    @classmethod
    def system(cls, text: str) -> Message:
        """Build a system turn from plain text."""
        return cls(role="system", parts=[TextPart(text)])

    @classmethod
    def from_tool_results(cls, results: Sequence[ToolResultPart]) -> Message:
        """Build the single tool turn answering a batch of parallel calls.

        Named ``from_tool_results`` rather than ``tool_results`` so it does not
        shadow the same-named property below.
        """
        return cls(role="tool", parts=list(results))

    @property
    def text(self) -> str:
        """Every :class:`TextPart` concatenated in order; thinking is excluded."""
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Tool calls requested in this turn, in the order the model emitted them."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def tool_results(self) -> list[ToolResultPart]:
        """Tool results carried by this turn."""
        return [part for part in self.parts if isinstance(part, ToolResultPart)]

    @property
    def has_tool_calls(self) -> bool:
        """Whether the loop must dispatch tools before asking the model again."""
        return any(isinstance(part, ToolCallPart) for part in self.parts)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one request, accumulable across a whole session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        """Billable total. Cached and reasoning tokens are subsets already
        counted in prompt/completion, so they are not added again."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The wire-level description of a tool, i.e. what a provider serialises.

    Deliberately decoupled from the executable tool object so the registry can
    advertise a subset of tools, or rewrite a schema, without touching runtime
    behaviour.
    """

    name: str
    description: str
    input_schema: dict[str, object]


class RiskLevel(enum.IntEnum):
    """How much damage running a tool could do.

    An :class:`~enum.IntEnum` so approval policies can compare with ``>=``.
    """

    SAFE = 0
    """Read-only; runs without asking."""

    MUTATING = 1
    """Writes files, installs packages, or otherwise mutates state."""

    DANGEROUS = 2
    """Destructive or irreversible; requires explicit typed approval."""

