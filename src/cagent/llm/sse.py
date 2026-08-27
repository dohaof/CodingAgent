"""A hand-written Server-Sent Events reader.

Every streaming endpoint this agent talks to frames its output as SSE, so the
parser lives here rather than inside a vendor SDK. It follows the WHATWG event
stream rules closely enough for real APIs: ``data:`` fields accumulate and are
joined with newlines, a blank line dispatches the buffered event, ``event:``
names are preserved (the Anthropic wire dispatches on them), lines beginning
with ``:`` are comments, and one optional space after the colon is stripped.

Two deliberate departures from the browser spec, both because these streams are
read once and never reconnected:

* An event whose ``data`` buffer is empty is still dispatched when it carried an
  ``event:`` name, so keep-alives like ``event: ping`` remain visible.
* ``id:`` and ``retry:`` are ignored instead of updating reconnection state.

The OpenAI ``[DONE]`` sentinel is surfaced as an ordinary event so the caller
decides what it means; :func:`iter_json_events` is the layer that drops it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..errors import ResponseParseError

__all__ = [
    "DONE_SENTINEL",
    "SSEEvent",
    "iter_json_events",
    "iter_sse",
]

DONE_SENTINEL = "[DONE]"
"""Terminator the OpenAI wire sends in place of a final JSON payload."""

DEFAULT_EVENT = "message"
"""Event name assumed when a block omits ``event:``, per the SSE spec."""

_EXCERPT_CHARS = 240


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One dispatched event block: its name and its joined data payload."""

    event: str
    data: str


def _excerpt(text: str) -> str:
    """Shorten a payload for an error message without losing its shape."""
    flattened = " ".join(text.split())
    if len(flattened) <= _EXCERPT_CHARS:
        return flattened
    return f"{flattened[:_EXCERPT_CHARS]}..."


def iter_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse an iterable of lines into dispatched :class:`SSEEvent` blocks.

    Line endings are stripped, so this accepts ``httpx``'s ``iter_lines()``
    output as well as a list of literal lines in a test. A trailing block that
    the server never terminated with a blank line is still yielded at the end of
    the stream, because a closed connection is as final as a blank line.
    """
    event_name = ""
    data_lines: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")

        if not line:
            if data_lines or event_name:
                yield SSEEvent(event=event_name or DEFAULT_EVENT, data="\n".join(data_lines))
            event_name = ""
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            # A bare field name carries the empty string as its value.
            field, value = line, ""
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if data_lines or event_name:
        yield SSEEvent(event=event_name or DEFAULT_EVENT, data="\n".join(data_lines))


def iter_json_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, object]]]:
    """Yield ``(event_name, payload)`` for every JSON-bearing event in a stream.

    Empty data blocks and the ``[DONE]`` sentinel are skipped, since neither
    carries model output.

    Raises:
        ResponseParseError: if a payload is not valid JSON, or decodes to
            something other than an object. The excerpt kept on the error is
            whitespace-collapsed and truncated so it can be logged or quoted
            back to the model safely.
    """
    for event in iter_sse(lines):
        data = event.data.strip()
        if not data or data == DONE_SENTINEL:
            continue

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                f"Stream event {event.event!r} carried malformed JSON: {exc.msg} "
                f"at position {exc.pos}.",
                _excerpt(data),
            ) from exc

        if not isinstance(payload, dict):
            raise ResponseParseError(
                f"Stream event {event.event!r} decoded to {type(payload).__name__}, "
                f"expected a JSON object.",
                _excerpt(data),
            )

        yield event.event, payload
