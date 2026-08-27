"""The provider seam: a neutral streaming event model and the provider ABC.

This module is the Adapter boundary that makes the rest of the agent
model-agnostic. Above it nothing knows whether a response arrived as OpenAI
``choices[].delta`` fragments or as Anthropic ``content_block_delta`` events;
below it each wire adapter translates its vendor payloads into the
:data:`StreamEvent` union defined here. No vendor field name, role string, or
content-block type may appear above this line.

The event vocabulary is deliberately small - text, thinking, tool-call framing,
usage, and a terminator - because every concept added here has to be expressible
by every wire.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import httpx

from ..config import AgentConfig
from ..errors import UserAbort
from ..types import (
    ContentPart,
    FinishReason,
    Message,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolSpec,
    Usage,
)

__all__ = [
    "CompletionResult",
    "LLMProvider",
    "StreamEvent",
    "StreamFinished",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallAccumulator",
    "ToolCallArgsDelta",
    "ToolCallStarted",
    "UsageReport",
]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of user-visible assistant prose."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A fragment of a reasoning trace.

    Kept separate from :class:`TextDelta` so the UI can style or hide it and the
    context manager can drop it from history without touching the answer.
    """

    text: str

    signature: str | None = None
    """Opaque provider blob, echoed back verbatim on the next request when set."""


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """A new tool call appeared in the stream.

    ``index`` is the call's position within this response and is the only field
    guaranteed to be present from the start; ``id`` and ``name`` may arrive empty
    and be completed by a later event.
    """

    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallArgsDelta:
    """A fragment of one tool call's argument JSON.

    Providers stream arguments as unframed text, so a fragment is usually not
    valid JSON on its own. Accumulation belongs to
    :class:`ToolCallAccumulator`.
    """

    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class UsageReport:
    """Token accounting for the request, when the provider reports it."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class StreamFinished:
    """Terminator carrying why generation stopped."""

    reason: FinishReason


StreamEvent = (
    TextDelta | ThinkingDelta | ToolCallStarted | ToolCallArgsDelta | UsageReport | StreamFinished
)
"""Everything a provider may emit. Exhaustive: consumers may match on it."""


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """One fully drained response.

    ``message`` is the assembled assistant turn, ready to append to the
    transcript. ``latency_s`` is wall-clock time for the whole stream, so it
    includes generation, not just time to first byte.
    """

    message: Message
    finish_reason: FinishReason
    usage: Usage
    latency_s: float


@dataclass(slots=True)
class _CallSlot:
    """Mutable accumulation state for one indexed tool call."""

    index: int
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


class ToolCallAccumulator:
    """Folds tool-call events into finished :class:`ToolCallPart`s.

    Written as an explicit state machine because real streams are messier than
    the documented shape:

    * Slots are keyed by ``index``, not by arrival order, since providers may
      interleave fragments from parallel calls.
    * A name can arrive split across chunks, or on a later event than the one
      that opened the slot. Names concatenate; ids do not, because an id is
      always sent whole and a repeat is a resend, not a continuation.
    * Empty arguments mean no arguments: ``""`` becomes ``{}``.
    * Invalid JSON never raises. The part is produced with ``arguments={}`` and
      the raw text preserved in ``raw_arguments``, and the index is listed in
      :attr:`malformed` so the engine can feed the model a repair prompt instead
      of losing the turn.
    """

    __slots__ = ("_slots", "malformed")

    def __init__(self) -> None:
        self._slots: dict[int, _CallSlot] = {}
        self.malformed: list[int] = []

    def feed(self, event: StreamEvent) -> None:
        """Absorb one event. Events other than tool-call events are ignored."""
        if isinstance(event, ToolCallStarted):
            slot = self._slots.get(event.index)
            if slot is None:
                slot = _CallSlot(index=event.index)
                self._slots[event.index] = slot
            if event.id:
                slot.id = event.id
            if event.name:
                slot.name += event.name
        elif isinstance(event, ToolCallArgsDelta):
            slot = self._slots.get(event.index)
            if slot is None:
                # Arguments before any start event: open the slot anyway rather
                # than dropping fragments a strict provider never re-sends.
                slot = _CallSlot(index=event.index)
                self._slots[event.index] = slot
            if event.delta:
                slot.args.append(event.delta)

    @property
    def pending(self) -> int:
        """How many calls are currently accumulating."""
        return len(self._slots)

    def finish(self) -> list[ToolCallPart]:
        """Materialise every accumulated call, ordered by index.

        Recomputes :attr:`malformed` from scratch, so calling this twice on the
        same state yields the same answer.
        """
        self.malformed = []
        parts: list[ToolCallPart] = []

        for index in sorted(self._slots):
            slot = self._slots[index]
            raw = "".join(slot.args).strip()
            arguments: dict[str, object] = {}
            raw_arguments = ""

            if raw:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    self.malformed.append(index)
                    raw_arguments = raw
                else:
                    if isinstance(decoded, dict):
                        arguments = decoded
                    else:
                        # Valid JSON of the wrong shape is as unusable as broken
                        # JSON, and the repair path is identical.
                        self.malformed.append(index)
                        raw_arguments = raw

            parts.append(
                ToolCallPart(
                    id=slot.id or f"call_{index}",
                    name=slot.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
        return parts

    def reset(self) -> None:
        """Drop all state, ready for the next response."""
        self._slots.clear()
        self.malformed = []


class LLMProvider(ABC):
    """Base class for wire adapters.

    Subclasses implement exactly one method, :meth:`stream`, and inherit
    assembly, timing, and token counting. The contract a subclass must honour:

    * **Exactly one** :class:`StreamFinished` is emitted, and it is last. The
      loop reads its reason to decide whether to dispatch tools or stop, so a
      stream that ends without one is a protocol error the adapter must convert
      into a ``StreamFinished("error")`` or raise a
      :class:`~cagent.errors.ProviderError`.
    * :class:`UsageReport` is optional and may arrive at most once. Providers
      that report usage mid-stream should emit it when they send it; callers
      accumulate whatever arrives and treat a missing report as zeroes.
    * Deltas are ordered. Text and thinking fragments arrive in generation order,
      and every :class:`ToolCallArgsDelta` for an index follows the
      :class:`ToolCallStarted` that opened it - except where a provider omits the
      start event, which :class:`ToolCallAccumulator` tolerates.
    * ``abort`` is polled between chunks. When it is set the adapter stops
      reading and emits ``StreamFinished("aborted")`` rather than raising, so a
      partial turn can still be recorded.

    An ``httpx.Client`` may be injected for tests (``MockTransport``) or to share
    a connection pool. An injected client is not closed by :meth:`close`, since
    the caller that created it owns its lifetime.
    """

    wire: ClassVar[str] = ""
    """Which request shape this adapter speaks; matches ``AgentConfig.wire``."""

    def __init__(self, config: AgentConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client(config)

    @staticmethod
    def _build_client(config: AgentConfig) -> httpx.Client:
        """Create the default client: no retries here, the retry layer owns those."""
        return httpx.Client(timeout=httpx.Timeout(config.request_timeout))

    @property
    def client(self) -> httpx.Client:
        """The HTTP client this provider issues requests on."""
        return self._client

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec] = (),
        abort: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Send one request and yield neutral events as the response arrives.

        Implementations must satisfy the guarantees in the class docstring.

        Raises:
            ProviderError: on transport failure or an undecodable response.
        """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec] = (),
        abort: threading.Event | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CompletionResult:
        """Drain :meth:`stream` into one assembled :class:`CompletionResult`.

        ``on_event`` is called for every event before it is folded in, which lets
        the UI render deltas live while this method still returns the finished
        message. Without that hook every caller wanting live output would have to
        reimplement assembly, and the two copies would drift.

        Part order in the assembled message is normalised - thinking first, then
        text, then tool calls - regardless of how the provider interleaved them,
        because that is the order every wire expects on the way back in.

        Raises:
            UserAbort: if ``abort`` is set before the stream produced anything.
            ProviderError: propagated from :meth:`stream`.
        """
        started = time.perf_counter()
        thinking: list[str] = []
        signature: str | None = None
        text: list[str] = []
        calls = ToolCallAccumulator()
        usage = Usage()
        reason: FinishReason = "error"
        saw_finish = False

        for event in self.stream(messages, system=system, tools=tools, abort=abort):
            if on_event is not None:
                on_event(event)

            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, ThinkingDelta):
                thinking.append(event.text)
                if event.signature:
                    signature = event.signature
            elif isinstance(event, ToolCallStarted | ToolCallArgsDelta):
                calls.feed(event)
            elif isinstance(event, UsageReport):
                usage = usage + event.usage
            elif isinstance(event, StreamFinished):
                reason = event.reason
                saw_finish = True

        latency = time.perf_counter() - started
        aborted = abort is not None and abort.is_set()
        parts: list[ContentPart] = []

        if thinking:
            parts.append(ThinkingPart("".join(thinking), signature))
        joined = "".join(text)
        if joined:
            parts.append(TextPart(joined))
        tool_parts = calls.finish()
        parts.extend(tool_parts)

        if not saw_finish:
            if aborted:
                reason = "aborted"
            elif not parts:
                raise UserAbort("Provider stream ended without producing any content.")
            else:
                # Content but no terminator: trust the content and let the loop
                # decide from the parts rather than discarding a whole turn.
                reason = "tool_calls" if tool_parts else "stop"

        return CompletionResult(
            message=Message.assistant(*parts),
            finish_reason=reason,
            usage=usage,
            latency_s=latency,
        )

    def count_tokens(
        self,
        messages: Sequence[Message],
        *,
        system: str = "",
        tools: Sequence[ToolSpec] = (),
    ) -> int:
        """Estimate the token cost of a request, for budgeting only.

        Imported inside the body because :mod:`cagent.llm.tokens` is free to
        import provider types without creating a cycle.
        """
        from .tokens import estimate_messages

        return estimate_messages(
            messages,
            system=system,
            tools=tools,
            model=self.config.resolved_model,
        )

    def close(self) -> None:
        """Close the client if this provider created it. Idempotent."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

