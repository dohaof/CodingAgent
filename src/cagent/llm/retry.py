"""Retry policy and HTTP error classification, independent of any transport.

Nothing here imports ``httpx``: :func:`with_retries` takes a plain callable and
:func:`classify_http_error` takes a status code, a body, and optional headers.
That keeps the retry rules testable without a socket, and lets the same policy
wrap a streaming request, a non-streaming one, or a token-count probe.

Only two failures are retried - :class:`~cagent.errors.RateLimitError` and
:class:`~cagent.errors.TransientProviderError`. Retrying an
:class:`~cagent.errors.AuthError` cannot succeed, retrying a
:class:`~cagent.errors.ContextOverflowError` needs the transcript compacted
first, and retrying a :class:`~cagent.errors.ResponseParseError` would replay a
request the model already answered badly. All three are re-raised immediately.

Backoff is full jitter: ``random() * min(max_delay, base * 2**attempt)``. Jitter
matters because a rate limit usually hits every concurrent request at once, and
an unjittered schedule would send them all back in lockstep.
"""

from __future__ import annotations

import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from ..errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
    UserAbort,
)

__all__ = [
    "RETRYABLE",
    "RetryPolicy",
    "classify_http_error",
    "with_retries",
]

T = TypeVar("T")

RETRYABLE: tuple[type[ProviderError], ...] = (RateLimitError, TransientProviderError)
"""The only errors :func:`with_retries` will attempt again."""

_CONTEXT_SIGNATURES: tuple[str, ...] = (
    "context length",
    "context_length",
    "context window",
    "context_window",
    "maximum context",
    "max_tokens",
    "too many tokens",
    "too long",
    "prompt is too long",
    "reduce the length",
    "input length",
    "exceeds the maximum",
    "string too long",
)
"""Substrings vendors use for an over-long prompt returned as a 400."""

_TOKEN_COUNTS = re.compile(r"(\d[\d,]{2,})\s*tokens")
_RETRY_AFTER_BODY = re.compile(
    r"retry[-_ ]?after[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RETRY_SECONDS = re.compile(
    r"(?:try again in|retry in)\s+(\d+(?:\.\d+)?)\s*(m?s|seconds?|minutes?)",
    re.IGNORECASE,
)
_EXCERPT_CHARS = 300
_ABORT_POLL_SECONDS = 0.25
"""Longest a backoff wait goes without noticing an abort flag."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to retry and how long to wait between attempts.

    ``max_retries`` counts retries, not total attempts: ``0`` means try once.
    ``jitter`` is the fraction of the computed delay left to chance, so ``0.0``
    makes the schedule deterministic for tests and ``1.0`` is pure full jitter.
    """

    max_retries: int = 4
    base_delay: float = 0.6
    max_delay: float = 20.0
    jitter: float = 0.3

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Seconds to wait before ``attempt`` (0-based, so 0 is the first retry).

        A server-supplied ``retry_after`` wins whenever it is larger than the
        computed backoff - the provider knows when its own window reopens - but
        is still capped by ``max_delay`` so a hostile or mistaken header cannot
        stall the agent indefinitely.
        """
        exponential = self.base_delay * (2.0 ** max(0, attempt))
        capped = min(self.max_delay, exponential)
        floor = capped * (1.0 - self.jitter)
        delay = floor + (capped - floor) * random.random()
        if retry_after is not None and retry_after > delay:
            delay = min(float(retry_after), self.max_delay)
        return max(0.0, delay)


def _excerpt(body: str) -> str:
    """Collapse and truncate a response body for an exception message."""
    flattened = " ".join(body.split())
    if len(flattened) <= _EXCERPT_CHARS:
        return flattened
    return f"{flattened[:_EXCERPT_CHARS]}..."


def _parse_retry_after(body: str, headers: Mapping[str, str] | None) -> float | None:
    """Extract a retry delay from headers first, then the body.

    Handles the header's delay-seconds form, a ``retry_after`` field in a JSON
    error body, and the prose form vendors use ("try again in 1.5s").
    """
    if headers:
        lowered = {key.lower(): value for key, value in headers.items()}
        for name in ("retry-after", "x-ratelimit-reset-after", "retry-after-ms"):
            raw = lowered.get(name)
            if not raw:
                continue
            try:
                value = float(str(raw).strip())
            except ValueError:
                continue
            return value / 1000.0 if name.endswith("-ms") else value

    match = _RETRY_AFTER_BODY.search(body)
    if match:
        return float(match.group(1))

    prose = _RETRY_SECONDS.search(body)
    if prose:
        value = float(prose.group(1))
        unit = prose.group(2).lower()
        if unit == "ms":
            return value / 1000.0
        if unit.startswith("minute"):
            return value * 60.0
        return value
    return None


def _context_tokens(body: str) -> tuple[int | None, int | None]:
    """Pull (required, window) token counts out of an overflow message.

    Vendors word this inconsistently but almost always print the window before
    the request size, e.g. "maximum context length is 128000 tokens, however you
    requested 131000 tokens". Two numbers are read in that order; a single number
    is treated as the window, which is the more useful of the two for compaction.
    """
    numbers = [int(raw.replace(",", "")) for raw in _TOKEN_COUNTS.findall(body)]
    if len(numbers) >= 2:
        return numbers[1], numbers[0]
    if len(numbers) == 1:
        return None, numbers[0]
    return None, None


def classify_http_error(
    status: int,
    body: str,
    headers: Mapping[str, str] | None = None,
) -> ProviderError:
    """Map an HTTP failure onto the error family that decides what happens next.

    * 401, 403 -> :class:`~cagent.errors.AuthError` (fatal)
    * 429 -> :class:`~cagent.errors.RateLimitError` (retried, honouring Retry-After)
    * 400, 413, 422 carrying a context/length signature ->
      :class:`~cagent.errors.ContextOverflowError` (compact, then retry)
    * 408, 409, 425, 5xx -> :class:`~cagent.errors.TransientProviderError` (retried)
    * anything else -> :class:`~cagent.errors.ProviderError` (fatal)

    A 429 is checked before the overflow signatures: a rate limit body often
    mentions token limits too, and waiting is the correct response to it.
    """
    excerpt = _excerpt(body)
    detail = f" Body: {excerpt}" if excerpt else ""

    if status in (401, 403):
        return AuthError(f"Provider rejected the credentials (HTTP {status}).{detail}")

    if status == 429:
        return RateLimitError(
            f"Provider rate limit hit (HTTP 429).{detail}",
            _parse_retry_after(body, headers),
        )

    if status in (400, 413, 422):
        lowered = body.lower()
        if any(signature in lowered for signature in _CONTEXT_SIGNATURES):
            required, window = _context_tokens(body)
            return ContextOverflowError(
                f"Request exceeded the model's context window (HTTP {status}).{detail}",
                required_tokens=required,
                window_tokens=window,
            )

    if status in (408, 409, 425) or 500 <= status < 600:
        return TransientProviderError(
            f"Provider returned a retryable HTTP {status}.{detail}"
        )

    return ProviderError(f"Provider request failed with HTTP {status}.{detail}")


def _wait(
    delay: float,
    *,
    sleep: Callable[[float], None],
    abort: threading.Event | None,
) -> None:
    """Sleep for ``delay``, in short slices when an abort flag is watched.

    Always goes through the injected ``sleep`` rather than ``Event.wait`` so a
    test that stubs sleeping stays instant even when it passes an abort event.
    """
    if abort is None:
        sleep(delay)
        return
    remaining = delay
    while remaining > 0.0 and not abort.is_set():
        slice_seconds = min(_ABORT_POLL_SECONDS, remaining)
        sleep(slice_seconds)
        remaining -= slice_seconds


def with_retries(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    abort: threading.Event | None = None,
) -> T:
    """Call ``fn``, retrying retryable provider errors under ``policy``.

    ``fn`` takes no arguments so the caller closes over its own request state;
    it must be safe to invoke more than once. ``on_retry`` receives
    ``(attempt, error, delay)`` before each wait, for logging or UI. ``sleep`` is
    injected so tests advance time without waiting.

    ``abort`` is checked before the first attempt and again around every wait, so
    a Ctrl-C during backoff is noticed immediately instead of after the delay.

    Raises:
        UserAbort: if ``abort`` is set before an attempt or during a wait.
        Exception: the last error raised by ``fn`` once retries are exhausted,
            and any non-retryable error immediately.
    """
    attempts = max(0, policy.max_retries) + 1
    last: Exception | None = None

    for attempt in range(attempts):
        if abort is not None and abort.is_set():
            raise UserAbort("Run aborted before the provider request was sent.")
        try:
            return fn()
        except RETRYABLE as exc:
            last = exc
            if attempt == attempts - 1:
                break
            retry_after = getattr(exc, "retry_after", None)
            delay = policy.delay_for(attempt, retry_after=retry_after)
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            _wait(delay, sleep=sleep, abort=abort)
            if abort is not None and abort.is_set():
                raise UserAbort("Run aborted while waiting to retry.") from exc

    if last is None:  # unreachable: the loop only breaks after a retryable error
        raise ProviderError("Retry loop ended without an outcome.")
    raise last
