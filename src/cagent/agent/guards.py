"""Stopping conditions: why the loop is not a ``while True``.

An agent that decides its own next action can fail by never stopping, and the
failure is expensive rather than loud — it burns tokens looking productive. Two
independent guards bound a run:

* a token budget, because cost is the resource the user actually feels;
* repetition detection, for the common live-lock where the model retries an
  identical failing call forever.

Repetition is handled in two stages rather than one. A repeat first earns a
nudge — a tool result telling the model it is looping and what to try instead —
because the usual cause is a model that cannot see its own pattern, and one
sentence of feedback is often enough. Only an unbroken streak past the limit
stops the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import AgentConfig
from ..errors import RepetitionDetected, TokenBudgetExceeded
from ..types import ToolCallPart

__all__ = ["LoopGuard", "call_signature"]

_NUDGE_AT = 2
"""Identical consecutive calls that trigger a warning rather than a stop."""


def call_signature(call: ToolCallPart) -> str:
    """A stable key for "the same call again".

    Arguments are serialised with sorted keys so that a model re-emitting the
    same call with its JSON fields reordered still counts as a repeat.
    """
    try:
        arguments = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        arguments = call.raw_arguments
    return f"{call.name}({arguments})"


@dataclass(slots=True)
class LoopGuard:
    """Tracks a run against its limits.

    Raises rather than returning a flag, because every limit here is a reason to
    unwind the loop rather than something a caller might reasonably ignore. The
    exceptions are :class:`~cagent.errors.LoopGuardError` subclasses, which the
    engine catches to end the run cleanly with an explanation.
    """

    config: AgentConfig
    steps: int = 0
    tokens_used: int = 0
    _last_signature: str | None = None
    _repeats: int = 0
    _nudged: set[str] = field(default_factory=set)

    def before_step(self) -> None:
        """Account for one more model request.

        Raises:
            TokenBudgetExceeded: If the token budget is already spent.
        """
        budget = self.config.token_budget
        if budget is not None and self.tokens_used >= budget:
            raise TokenBudgetExceeded(self.tokens_used, budget)
        self.steps += 1

    def add_tokens(self, count: int) -> None:
        """Record tokens spent by a completed request."""
        self.tokens_used += max(count, 0)

    def check_call(self, call: ToolCallPart) -> str | None:
        """Register a tool call and report whether the model is looping.

        Returns:
            A nudge to hand back to the model as a tool result, or ``None`` when
            the call is not a concerning repeat.

        Raises:
            RepetitionDetected: If the same call has now repeated past the limit,
                which means the nudge was already tried and ignored.
        """
        signature = call_signature(call)
        if signature != self._last_signature:
            self._last_signature = signature
            self._repeats = 1
            return None

        self._repeats += 1
        limit = max(self.config.max_repeated_calls, 1)
        if self._repeats > limit:
            raise RepetitionDetected(call.name, self._repeats)
        if self._repeats >= _NUDGE_AT and signature not in self._nudged:
            self._nudged.add(signature)
            return (
                f"This is call {self._repeats} of {call.name} with identical arguments. "
                "Repeating it will return the same result. Change the arguments, gather "
                "more information with a different tool, or explain what is blocking you."
            )
        return None

    def note_progress(self) -> None:
        """Reset repetition tracking after a genuinely different action.

        Called when a turn ends, so a call repeated in a later turn at the
        user's request is not mistaken for a live-lock.
        """
        self._last_signature = None
        self._repeats = 0
