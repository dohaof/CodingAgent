"""Indicative token prices, for showing what a session cost.

Published rates, in US dollars per million tokens, recorded here so a run can
report a figure rather than a token count nobody converts in their head. They
are indicative only: vendors change prices, discount cached input, and bill
differently per region, so an unknown model reports tokens and no dollar amount
instead of guessing.

Matching is by prefix, because deployed model names carry dated suffixes
(``deepseek-chat``, ``gpt-4o-mini-2024-07-18``) that would otherwise each need
their own row.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Price", "estimate_cost", "price_for"]


@dataclass(frozen=True, slots=True)
class Price:
    """Dollars per million tokens."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None
    """Rate for prompt tokens served from cache, when the vendor discounts them."""


_PRICES: tuple[tuple[str, Price], ...] = (
    # Longest prefixes first: the first match wins, so a specific variant must
    # be listed before the family it belongs to.
    ("deepseek-reasoner", Price(0.55, 2.19, 0.14)),
    ("deepseek-chat", Price(0.27, 1.10, 0.07)),
    ("gpt-4o-mini", Price(0.15, 0.60, 0.075)),
    ("gpt-4o", Price(2.50, 10.00, 1.25)),
    ("gpt-4.1-mini", Price(0.40, 1.60, 0.10)),
    ("gpt-4.1", Price(2.00, 8.00, 0.50)),
    ("claude-sonnet-4", Price(3.00, 15.00, 0.30)),
    ("claude-opus-4", Price(15.00, 75.00, 1.50)),
    ("claude-haiku-4", Price(1.00, 5.00, 0.10)),
    ("claude-3-5-haiku", Price(0.80, 4.00, 0.08)),
    ("claude-3-5-sonnet", Price(3.00, 15.00, 0.30)),
    ("qwen-plus", Price(0.40, 1.20)),
    ("qwen-turbo", Price(0.05, 0.20)),
    ("kimi-k2", Price(0.60, 2.50)),
    ("moonshot-v1", Price(0.20, 0.80)),
)


def price_for(model: str) -> Price | None:
    """The price table entry for ``model``, or ``None`` when unlisted."""
    lowered = model.lower()
    for prefix, price in _PRICES:
        if lowered.startswith(prefix):
            return price
    return None


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float | None:
    """Estimate a run's cost in dollars, or ``None`` for an unlisted model.

    Cached prompt tokens are billed at the cached rate where one is known and
    are treated as a subset of ``prompt_tokens``, matching how both wires report
    them.
    """
    price = price_for(model)
    if price is None:
        return None

    cached = min(max(cached_tokens, 0), max(prompt_tokens, 0))
    fresh = max(prompt_tokens, 0) - cached
    cached_rate = (
        price.cached_input_per_m
        if price.cached_input_per_m is not None
        else price.input_per_m
    )

    return (
        fresh * price.input_per_m
        + cached * cached_rate
        + max(completion_tokens, 0) * price.output_per_m
    ) / 1_000_000
