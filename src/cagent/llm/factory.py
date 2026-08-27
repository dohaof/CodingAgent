"""Provider construction: one lookup from wire name to adapter class.

Kept separate from the adapters so importing a single wire module never drags
in the other, and separate from :mod:`~cagent.llm.base` so the ABC stays free
of subclass knowledge.
"""

from __future__ import annotations

import httpx

from ..config import AgentConfig
from ..errors import ConfigError
from .anthropic_wire import AnthropicProvider
from .base import LLMProvider
from .openai_wire import OpenAIProvider

__all__ = ["WIRE_IMPLEMENTATIONS", "build_provider"]

WIRE_IMPLEMENTATIONS: dict[str, type[LLMProvider]] = {
    OpenAIProvider.wire: OpenAIProvider,
    AnthropicProvider.wire: AnthropicProvider,
}
"""Every wire this build knows how to speak, keyed by ``AgentConfig.wire``."""


def build_provider(config: AgentConfig, *, client: httpx.Client | None = None) -> LLMProvider:
    """Instantiate the adapter for ``config.resolved_wire``.

    ``client`` is passed through for tests and connection-pool sharing; the
    provider will not close an injected client.

    Raises:
        ConfigError: if no adapter implements the resolved wire.
    """
    wire = config.resolved_wire
    implementation = WIRE_IMPLEMENTATIONS.get(wire)
    if implementation is None:
        known = ", ".join(sorted(WIRE_IMPLEMENTATIONS))
        raise ConfigError(f"No provider implements wire {wire!r}. Known wires: {known}.")
    return implementation(config, client=client)
