"""Exception hierarchy for the agent.

Three families matter to control flow. :class:`ProviderError` decides whether a
request is retried, rewritten, or fatal. :class:`ToolError` never aborts the
loop: its :meth:`ToolError.as_model_feedback` string is written back into the
transcript so the model can self-correct. :class:`LoopGuardError` stops a
runaway session.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "ApprovalDeniedError",
    "AuthError",
    "CagentError",
    "ConfigError",
    "ContextOverflowError",
    "LoopGuardError",
    "PathOutsideWorkspaceError",
    "ProviderError",
    "RateLimitError",
    "RepetitionDetected",
    "ResponseParseError",
    "ToolArgumentError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "TokenBudgetExceeded",
    "TransientProviderError",
    "UserAbort",
]


class CagentError(Exception):
    """Base class for everything this package raises deliberately."""


class ConfigError(CagentError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ProviderError(CagentError):
    """Base class for LLM transport and protocol failures."""


class AuthError(ProviderError):
    """Credentials were rejected. Never retried."""


class RateLimitError(ProviderError):
    """Provider asked us to slow down.

    ``retry_after`` is the server-advertised delay in seconds when the response
    supplied one; otherwise the caller falls back to its own backoff schedule.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class TransientProviderError(ProviderError):
    """Timeout, connection reset, or 5xx. Safe to retry with backoff."""


class ContextOverflowError(ProviderError):
    """The request exceeded the model's context window.

    Carries the sizes when the provider reports them, so the context manager can
    compact by a known amount instead of guessing.
    """

    def __init__(
        self,
        message: str,
        *,
        required_tokens: int | None = None,
        window_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.required_tokens = required_tokens
        self.window_tokens = window_tokens


class ResponseParseError(ProviderError):
    """A response could not be decoded into the internal message model.

    ``raw`` keeps the offending payload so a repair prompt can quote it.
    """

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.raw = raw


class ToolError(CagentError):
    """Base class for tool-layer failures.

    These are expected outcomes, not crashes: the dispatcher catches them and
    feeds :meth:`as_model_feedback` back to the model as a tool result. ``hint``
    is therefore written for the model, telling it what to do differently.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def as_model_feedback(self) -> str:
        """Render the transcript-visible text for this failure."""
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ToolNotFoundError(ToolError):
    """The model called a tool that is not registered."""

    def __init__(self, name: str, available: Sequence[str] = ()) -> None:
        hint = f"Available tools: {', '.join(sorted(available))}." if available else None
        super().__init__(f"No tool named {name!r} is registered.", hint)
        self.name = name
        self.available = tuple(available)


class ToolArgumentError(ToolError):
    """Arguments were missing, of the wrong type, or failed schema validation."""


class ToolExecutionError(ToolError):
    """The tool ran but failed. ``exit_code`` is set for process-backed tools."""

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        *,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message, hint)
        self.exit_code = exit_code


class PathOutsideWorkspaceError(ToolError):
    """A path resolved outside the workspace root and the sandbox refused it."""

    def __init__(self, path: Path | str, workspace: Path | str) -> None:
        super().__init__(
            f"Path {str(path)!r} resolves outside the workspace root {str(workspace)!r}.",
            "Use a path inside the workspace, written relative to its root.",
        )
        self.path = str(path)
        self.workspace = str(workspace)


class ApprovalDeniedError(ToolError):
    """The user declined to approve a mutating or dangerous call."""

    def __init__(self, tool_name: str, reason: str | None = None) -> None:
        detail = f" Reason: {reason}" if reason else ""
        super().__init__(
            f"The user denied permission to run {tool_name!r}.{detail}",
            "Do not retry this call. Propose an alternative or ask the user how to proceed.",
        )
        self.tool_name = tool_name
        self.reason = reason


class LoopGuardError(CagentError):
    """Base class for the guards that stop a runaway agentic loop."""


class TokenBudgetExceeded(LoopGuardError):
    """Cumulative token spend passed the configured budget."""

    def __init__(self, used_tokens: int, budget_tokens: int) -> None:
        super().__init__(
            f"Token budget exhausted: used {used_tokens} of {budget_tokens}."
        )
        self.used_tokens = used_tokens
        self.budget_tokens = budget_tokens


class RepetitionDetected(LoopGuardError):
    """The same tool call repeated identically, so the loop is not progressing."""

    def __init__(self, tool_name: str, count: int) -> None:
        super().__init__(
            f"Tool {tool_name!r} was called with identical arguments {count} times in a row."
        )
        self.tool_name = tool_name
        self.count = count


class UserAbort(CagentError):
    """The user interrupted the run, e.g. with Ctrl-C."""
