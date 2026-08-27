"""Reporting what a session cost, when the rates are known.

No price table ships with this project, deliberately. Vendors change prices,
rename models, and bill cached input differently, so a built-in table is wrong
within months — and a stale *price* is worse than a missing one, because it
reports a confident dollar figure that is simply false. Token counts, which the
provider reports and which never go stale, are always shown.

To see costs, state your own rates in ``.cagent.toml``. Prefix matching means a
dated deployment name (``…-2024-07-18``) needs no separate row:

    [cagent.prices."your-model"]
    input_per_m = 0.27
    output_per_m = 1.10
    cached_input_per_m = 0.07   # optional

Rates are US dollars per million tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["Price", "estimate_cost", "parse_prices", "price_for"]


@dataclass(frozen=True, slots=True)
class Price:
    """Dollars per million tokens."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None
    """Rate for prompt tokens served from cache, when the vendor discounts them."""


def parse_prices(raw: Mapping[str, object]) -> dict[str, Price]:
    """Build a price table from configuration.

    Malformed entries are skipped rather than fatal: a typo in an optional
    cosmetic setting must not stop the agent from doing the user's work.
    """
    table: dict[str, Price] = {}
    for name, entry in raw.items():
        if not isinstance(entry, Mapping):
            continue
        try:
            cached = entry.get("cached_input_per_m")
            table[str(name).lower()] = Price(
                input_per_m=float(entry["input_per_m"]),
                output_per_m=float(entry["output_per_m"]),
                cached_input_per_m=None if cached is None else float(cached),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return table


def price_for(model: str, prices: Mapping[str, Price] | None = None) -> Price | None:
    """The rate configured for ``model``, or ``None`` when none is.

    Matching is by prefix, longest first, so ``gpt-4o`` and ``gpt-4o-mini`` can
    coexist and the more specific one wins.
    """
    if not prices:
        return None
    lowered = model.lower()
    for prefix in sorted(prices, key=len, reverse=True):
        if lowered.startswith(prefix):
            return prices[prefix]
    return None


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    prices: Mapping[str, Price] | None = None,
) -> float | None:
    """A run's cost in dollars, or ``None`` when no rate is configured.

    Cached prompt tokens are billed at the cached rate where one is given and
    are treated as a subset of ``prompt_tokens``, matching how both wires report
    them.
    """
    price = price_for(model, prices)
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
