"""The LLM transport layer: one neutral streaming contract, many wires.

:mod:`~cagent.llm.base` defines the seam - the :data:`~cagent.llm.base.StreamEvent`
union and the :class:`~cagent.llm.base.LLMProvider` ABC - and the wire adapters
translate vendor payloads into it. Supporting modules are imported directly by
the code that needs them rather than re-exported here:

* :mod:`cagent.llm.sse` - a hand-written Server-Sent Events reader.
* :mod:`cagent.llm.tokens` - token estimation, exact with ``tiktoken`` and
  heuristic without it.
* :mod:`cagent.llm.retry` - backoff policy and HTTP error classification.
* :mod:`cagent.llm.factory` - ``build_provider``, added with the adapters.

Only :mod:`~cagent.llm.base` is pulled in at import time, so importing this
package costs one module plus ``httpx``.
"""

from __future__ import annotations

from .base import (
    CompletionResult,
    LLMProvider,
    StreamEvent,
    StreamFinished,
    TextDelta,
    ThinkingDelta,
    ToolCallAccumulator,
    ToolCallArgsDelta,
    ToolCallStarted,
    UsageReport,
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
