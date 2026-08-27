"""Token accounting, with graceful degradation when no tokeniser is installed.

Every number produced here is an **estimate used for budgeting only** - deciding
when to compact the transcript, how much of a tool result to keep, whether a
request will fit the window. It is never a billing figure; read
:class:`~cagent.types.Usage` off a provider response for that.

Two backends. When ``tiktoken`` is importable its encoder is exact for OpenAI
family models and close enough elsewhere. When it is absent a hand-written
heuristic takes over: CJK codepoints count as roughly one token each, and the
remaining text is split on word and punctuation boundaries rather than divided by
a flat characters-per-token constant. That matters for this agent because source
code is punctuation-dense - ``len(text) / 4`` underestimates a minified JSON blob
and overestimates prose, while the boundary split stays within a useful margin on
both.

Structural overhead is added per message, per tool call, and per tool spec: the
wire format spends tokens on roles, delimiters, and JSON scaffolding that the
text itself does not account for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Sequence
from collections.abc import Set as AbstractSet
from functools import lru_cache
from math import ceil
from typing import Literal, Protocol

from ..types import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)

__all__ = [
    "MESSAGE_OVERHEAD_TOKENS",
    "REQUEST_OVERHEAD_TOKENS",
    "TOOL_CALL_OVERHEAD_TOKENS",
    "TOOL_RESULT_OVERHEAD_TOKENS",
    "TOOL_SPEC_OVERHEAD_TOKENS",
    "estimate_message",
    "estimate_messages",
    "estimate_text",
    "estimate_tools",
    "tiktoken_available",
]

MESSAGE_OVERHEAD_TOKENS = 4
"""Role marker plus message delimiters, per message."""

TOOL_CALL_OVERHEAD_TOKENS = 8
"""Call id, name framing, and the JSON envelope around one tool call."""

TOOL_RESULT_OVERHEAD_TOKENS = 4
"""Call id and result framing, per tool result."""

TOOL_SPEC_OVERHEAD_TOKENS = 8
"""Per-tool framing in the request's tool declaration array."""

REQUEST_OVERHEAD_TOKENS = 3
"""Fixed per-request priming the provider adds around the whole transcript."""

_FALLBACK_ENCODING = "o200k_base"
_CHARS_PER_TOKEN_FACTOR = 0.3
"""Tokens per character inside an alphanumeric run, from measured English and
identifier-heavy samples. Applied only after CJK and punctuation are counted."""

_TOKENISH = re.compile(
    r"(?P<word>[A-Za-z0-9_']+)|(?P<space>\s+)|(?P<punct>[^\sA-Za-z0-9_'])",
)

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2FFF),  # CJK radicals, Kangxi
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # Unified ideographs extension A
    (0x4E00, 0x9FFF),  # Unified ideographs
    (0xA960, 0xA97F),  # Hangul jamo extended A
    (0xAC00, 0xD7FF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # Halfwidth and fullwidth forms
    (0x20000, 0x2FA1F),  # Ideographic extensions B onward
)


class _Encoding(Protocol):
    """The one method this module needs from a tokeniser.

    Structurally matches ``tiktoken.Encoding.encode`` so the soft import needs no
    cast; ``disallowed_special=()`` is the only keyword this module passes.
    """

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = ...,
        disallowed_special: Literal["all"] | Collection[str] = ...,
    ) -> list[int]: ...


@lru_cache(maxsize=16)
def _encoder(model: str) -> _Encoding | None:
    """Return a cached encoder for ``model``, or ``None`` without ``tiktoken``.

    Cached per model name because building an encoder downloads and compiles a
    merge table, which is far too expensive to repeat per message.
    """
    try:
        import tiktoken
    except ImportError:
        return None

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            pass
    try:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)
    except (KeyError, ValueError, OSError):
        # No network on first use, or an unknown encoding name: fall back to the
        # heuristic rather than failing a request over an estimate.
        return None


def tiktoken_available() -> bool:
    """Whether an exact tokeniser backs the estimates in this process."""
    return _encoder("") is not None


def _is_cjk(char: str) -> bool:
    """Whether a codepoint is CJK, i.e. roughly one token on its own."""
    point = ord(char)
    return any(low <= point <= high for low, high in _CJK_RANGES)


def _heuristic_text(text: str) -> int:
    """Estimate tokens without a tokeniser.

    CJK codepoints are counted individually and replaced by a space so they do
    not fuse neighbouring words. What remains is split into alphanumeric runs
    (charged by length), single punctuation marks (one token each), and
    whitespace (free, since tokenisers attach it to the following word).
    """
    cjk_tokens = 0
    remainder: list[str] = []
    for char in text:
        if _is_cjk(char):
            cjk_tokens += 1
            remainder.append(" ")
        else:
            remainder.append(char)

    total = cjk_tokens
    for match in _TOKENISH.finditer("".join(remainder)):
        kind = match.lastgroup
        if kind == "word":
            total += max(1, ceil(len(match.group()) * _CHARS_PER_TOKEN_FACTOR))
        elif kind == "punct":
            total += 1
    return total


def estimate_text(text: str, *, model: str = "") -> int:
    """Estimate the token count of ``text`` for ``model``.

    Uses ``tiktoken`` when available, otherwise :func:`_heuristic_text`. Special
    tokens in the input are treated as ordinary text: user content and file
    contents routinely contain sequences like ``<|endoftext|>``, and an estimate
    must never raise on them.
    """
    if not text:
        return 0
    encoder = _encoder(model)
    if encoder is None:
        return _heuristic_text(text)
    return len(encoder.encode(text, disallowed_special=()))


def _json_size(value: object, *, model: str) -> int:
    """Estimate the tokens a value costs once serialised onto the wire."""
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(value)
    return estimate_text(rendered, model=model)


def _estimate_part(part: object, *, model: str) -> int:
    """Tokens for one content part, structural framing included."""
    if isinstance(part, TextPart):
        return estimate_text(part.text, model=model)
    if isinstance(part, ThinkingPart):
        total = estimate_text(part.text, model=model)
        if part.signature:
            total += estimate_text(part.signature, model=model)
        return total
    if isinstance(part, ToolCallPart):
        # Prefer raw_arguments when present: it is the exact text the provider
        # sent or will receive, so re-serialising the parsed dict would measure
        # a different string than the one billed.
        payload = (
            estimate_text(part.raw_arguments, model=model)
            if part.raw_arguments
            else _json_size(part.arguments, model=model)
        )
        return TOOL_CALL_OVERHEAD_TOKENS + estimate_text(part.name, model=model) + payload
    if isinstance(part, ToolResultPart):
        return TOOL_RESULT_OVERHEAD_TOKENS + estimate_text(part.content, model=model)
    return 0


def estimate_message(msg: Message, *, model: str = "") -> int:
    """Estimate one message, caching the result on the message itself.

    A set :attr:`~cagent.types.Message.token_estimate` is trusted and returned
    as is; the context manager clears it to ``None`` after rewriting parts, which
    is the signal to recompute. The cache is what keeps compaction from being
    quadratic: without it every pass would re-encode the whole transcript.
    """
    cached = msg.token_estimate
    if cached is not None:
        return cached

    total = MESSAGE_OVERHEAD_TOKENS + sum(_estimate_part(part, model=model) for part in msg.parts)
    msg.token_estimate = total
    return total


def estimate_tools(tools: Iterable[ToolSpec], *, model: str = "") -> int:
    """Estimate the cost of declaring ``tools`` on a request."""
    total = 0
    for tool in tools:
        total += (
            TOOL_SPEC_OVERHEAD_TOKENS
            + estimate_text(tool.name, model=model)
            + estimate_text(tool.description, model=model)
            + _json_size(tool.input_schema, model=model)
        )
    return total


def estimate_messages(
    messages: Sequence[Message],
    *,
    system: str = "",
    tools: Sequence[ToolSpec] = (),
    model: str = "",
) -> int:
    """Estimate a whole request: system prompt, tool declarations, transcript.

    This is the number the context manager compares against
    :attr:`~cagent.config.AgentConfig.compact_at_tokens`, and the one
    :meth:`~cagent.llm.base.LLMProvider.count_tokens` returns.
    """
    total = REQUEST_OVERHEAD_TOKENS
    if system:
        total += MESSAGE_OVERHEAD_TOKENS + estimate_text(system, model=model)
    if tools:
        total += estimate_tools(tools, model=model)
    return total + sum(estimate_message(msg, model=model) for msg in messages)

