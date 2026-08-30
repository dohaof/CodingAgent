"""The tool contract: injected context, approval requests, and the base class.

Two invariants hold the tool layer together.

First, a tool reaches for nothing globally. Everything it needs from its host —
the workspace root, the config, the approval callback, the progress sink, the
abort flag — arrives as a :class:`ToolContext` parameter, which is what makes
tools testable without a running agent.

Second, :meth:`BaseTool.invoke` never raises. Bad arguments, a refused path, an
unexpected ``KeyError`` deep in a helper: each becomes a :class:`ToolOutcome`
with ``is_error`` set, whose text the loop hands back to the model as a tool
result. A tool failure is an observation the model can act on, not a crash.
"""

from __future__ import annotations

import os
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..config import AgentConfig
from ..errors import PathOutsideWorkspaceError, ToolError
from ..types import RiskLevel, ToolSpec
from .schema import build_object_schema, parse_object

if TYPE_CHECKING:
    from ..agent.sandbox import SandboxSession

__all__ = [
    "ApprovalRequest",
    "BaseTool",
    "ToolContext",
    "ToolOutcome",
]

_TRACEBACK_TAIL_LINES = 6
"""How much of an unexpected traceback the model is shown."""


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One request for the user's permission to run a call.

    ``signature`` is a stable key for "always allow this": it must depend on the
    shape of the action and not on volatile detail, so repeating the same call
    hits the remembered decision.
    """

    tool: str
    risk: RiskLevel
    summary: str
    """One line, shown inline in the prompt."""

    detail: str | None = None
    """Expanded body, e.g. a rendered diff or the exact command."""

    signature: str = ""
    """Stable key for remembered approvals; defaults to the tool name."""

    always_prompt: bool = False
    """Require a fresh human answer even in ``full-auto`` mode.

    Used for copying an isolated workspace back to the real project: execution
    has already happened safely, but crossing that boundary is a separate act
    which a broad pre-approved command signature must not silently authorise.
    """


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The uniform return of every tool.

    ``content`` is the only field the model sees. ``display`` is for the human
    when a richer rendering exists, and ``metadata`` carries structured facts for
    the trace, so the transcript stays prose and the trace stays queryable.
    """

    content: str
    is_error: bool = False
    display: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def ok(cls, content: str, **kwargs: Any) -> ToolOutcome:
        """A successful outcome."""
        return cls(content=content, is_error=False, **kwargs)

    @classmethod
    def error(cls, content: str, **kwargs: Any) -> ToolOutcome:
        """A failed outcome the model is expected to react to."""
        kwargs.pop("is_error", None)
        return cls(content=content, is_error=True, **kwargs)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool needs from its host, injected rather than imported."""

    workspace: Path
    config: AgentConfig
    approve: Callable[[ApprovalRequest], bool]
    """Returns whether the call may proceed. May block on the user."""

    emit: Callable[[str], None]
    """Publishes a one-line progress update to the UI."""

    abort: threading.Event
    """Set when the user interrupts; long-running tools should poll it."""

    force_workspace_boundary: bool = False
    """Ignore ``allow_outside_workspace`` for an isolated snapshot."""

    sandbox: SandboxSession | None = None
    """Session snapshot used by isolated file and shell tools."""

    def resolve_path(self, raw: str) -> Path:
        """Resolve a model-supplied path and enforce the workspace sandbox.

        Relative paths are taken against the workspace root, ``~`` is expanded,
        and the result is fully resolved before the containment check, so that
        both ``../`` traversal and a symlink pointing out of the tree are caught.

        Args:
            raw: A path as the model wrote it, absolute or workspace-relative.

        Returns:
            An absolute, symlink-resolved path.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace and
                ``config.allow_outside_workspace`` is false.
        """
        text = os.path.expanduser(raw.strip()) if raw.strip() else raw
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate

        resolved = self._resolve(candidate)
        if self.config.allow_outside_workspace and not self.force_workspace_boundary:
            return resolved

        root = self._resolve(self.workspace)
        if resolved != root and not resolved.is_relative_to(root):
            raise PathOutsideWorkspaceError(raw, root)
        return resolved

    @staticmethod
    def _resolve(path: Path) -> Path:
        """Fully resolve ``path``, tolerating components that do not exist yet.

        ``strict=False`` is what lets a write tool name a file it is about to
        create while still resolving every symlink in the existing prefix.
        """
        try:
            return path.resolve()
        except OSError:
            return Path(os.path.normpath(str(path.absolute())))

    def rel(self, p: Path) -> str:
        """Render a path for display, workspace-relative where possible."""
        try:
            return p.relative_to(self.workspace).as_posix() or "."
        except ValueError:
            return str(p)


class BaseTool(ABC):
    """A single capability exposed to the model.

    Subclasses set the class variables and implement :meth:`run`. The
    schema advertised to the provider is derived from :attr:`Params`, so the
    arguments a tool declares and the arguments it receives cannot drift apart.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE
    parallel_safe: ClassVar[bool] = False
    """Whether independent calls may run concurrently in the same workspace."""
    Params: ClassVar[type]
    """The dataclass of model-supplied arguments."""

    _spec_cache: ClassVar[dict[str, ToolSpec]] = {}

    @classmethod
    def spec(cls) -> ToolSpec:
        """The wire description of this tool, built once per class.

        Raises:
            ToolArgumentError: If :attr:`Params` cannot be reflected into a
                schema. Raised at first use rather than silently shipping a
                wrong schema to the model.
        """
        key = f"{cls.__module__}.{cls.__qualname__}"
        cached = BaseTool._spec_cache.get(key)
        if cached is not None:
            return cached
        spec = ToolSpec(
            name=cls.name,
            description=cls.description.strip(),
            input_schema=build_object_schema(cls.Params),
        )
        BaseTool._spec_cache[key] = spec
        return spec

    def invoke(self, raw_args: Mapping[str, object], ctx: ToolContext) -> ToolOutcome:
        """Parse arguments, run the tool, and convert any failure to an outcome.

        This is the boundary that keeps the agent loop alive: nothing raised
        below it escapes. Argument errors and :class:`ToolError` subclasses
        become their model-facing feedback text; anything unexpected is reported
        with its type, message, and a short traceback tail, which is enough for
        the model to reason about without exposing the process environment.
        """
        params: Any
        try:
            params = parse_object(self.Params, raw_args)
        except ToolError as exc:
            return ToolOutcome.error(
                exc.as_model_feedback(),
                metadata={"tool": self.name, "phase": "arguments"},
            )

        try:
            return self.run(params, ctx)
        except ToolError as exc:
            return ToolOutcome.error(
                exc.as_model_feedback(),
                metadata={"tool": self.name, "phase": "run", "error": type(exc).__name__},
            )
        except Exception as exc:  # noqa: BLE001  # the loop must survive any tool
            return ToolOutcome.error(
                self._describe_crash(exc),
                metadata={"tool": self.name, "phase": "run", "error": type(exc).__name__},
            )

    @staticmethod
    def _describe_crash(exc: Exception) -> str:
        """Render an unexpected exception for the model.

        Only frames are shown, never locals or the environment, so a crash in a
        tool holding a credential cannot leak it into the transcript.
        """
        frames = traceback.format_tb(exc.__traceback__)
        tail = "".join(frames[-_TRACEBACK_TAIL_LINES:]).rstrip()
        head = f"The tool failed unexpectedly: {type(exc).__name__}: {exc}"
        if not tail:
            return head
        return f"{head}\nWhere it failed:\n{tail}"

    @abstractmethod
    def run(self, params: Any, ctx: ToolContext) -> ToolOutcome:
        """Do the work. Raise :class:`ToolError` for an expected failure.

        Args:
            params: An instance of :attr:`Params`, already validated.
            ctx: The injected host context.
        """

    def approval_request(self, params: Any, ctx: ToolContext) -> ApprovalRequest | None:
        """The permission prompt for this call, or ``None`` to run unattended.

        The default answers from :attr:`risk` alone. Tools with something
        concrete to show — a diff, a command line — should override and fill in
        ``detail`` and a ``signature`` narrower than the tool name.
        """
        if self.risk is RiskLevel.SAFE:
            return None
        return ApprovalRequest(
            tool=self.name,
            risk=self.risk,
            summary=f"Run {self.name}",
            detail=None,
            signature=self.name,
        )

    def preview(self, params: Any, ctx: ToolContext) -> str | None:
        """Text the UI may show before running, e.g. a diff. ``None`` if none."""
        return None
