"""The LLM transport layer: one neutral streaming contract, many wires.

:mod:`~cagent.llm.base` defines the seam - the :data:`~cagent.llm.base.StreamEvent`
union and the :class:`~cagent.llm.base.LLMProvider` ABC - and the wire adapters
translate vendor payloads into it:

* :mod:`cagent.llm.openai_wire` - the OpenAI Chat Completions dialect, spoken
  by most vendors.
* :mod:`cagent.llm.anthropic_wire` - the Anthropic Messages dialect.
* :mod:`cagent.llm.factory` - ``build_provider``, dispatching on the wire name.

Supporting modules are imported directly by the code that needs them rather
than re-exported here:

* :mod:`cagent.llm.sse` - a hand-written Server-Sent Events reader.
* :mod:`cagent.llm.tokens` - token estimation, exact with ``tiktoken`` and
  heuristic without it.
* :mod:`cagent.llm.retry` - backoff policy and HTTP error classification.
"""

from __future__ import annotations

from .anthropic_wire import AnthropicProvider
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
from .factory import WIRE_IMPLEMENTATIONS, build_provider
from .openai_wire import OpenAIProvider

__all__ = [
    "WIRE_IMPLEMENTATIONS",
    "AnthropicProvider",
    "CompletionResult",
    "LLMProvider",
    "OpenAIProvider",
    "StreamEvent",
    "StreamFinished",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallAccumulator",
    "ToolCallArgsDelta",
    "ToolCallStarted",
    "UsageReport",
    "build_provider",
]
