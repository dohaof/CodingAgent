"""Who decides whether a call runs.

The policy sits between a tool's own account of its risk and the user's
tolerance for being interrupted. It answers one question — may this call
proceed — and it answers it in one place, so the rule is auditable instead of
scattered through the tools.

Three modes trade safety against friction:

``suggest``
    Every mutation is confirmed. The cautious default for someone else's code.
``auto-edit``
    File edits inside the workspace run unattended, because they are reviewable
    afterwards — a diff is visible and version control can undo it. Shell
    commands still stop, because their effects reach outside the workspace and
    are not always reversible.
``full-auto``
    Only destructive commands stop. For a sandbox or a scratch repository.

No mode auto-approves :attr:`~cagent.types.RiskLevel.DANGEROUS`. A blanket
"always allow" is likewise refused for dangerous calls: the remembered key for
those is the command's full text, so consent is never extended from one
irreversible action to a different one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import AgentConfig
from ..tools.base import ApprovalRequest
from ..types import RiskLevel

__all__ = ["ApprovalPolicy", "Decision", "Prompter"]

Prompter = Callable[[ApprovalRequest], "Decision"]
"""Asks the user about one request. Supplied by the UI layer."""


@dataclass(frozen=True, slots=True)
class Decision:
    """A user's answer to one approval request."""

    approved: bool
    remember: bool = False
    """Apply this answer to every later request with the same signature."""

    abort: bool = False
    """The user wants the whole run to stop, not just this call."""


@dataclass(slots=True)
class ApprovalPolicy:
    """Decides, per call, whether to run, ask, or refuse.

    ``asked`` and ``auto_approved`` are kept for the end-of-run summary: a user
    who ran in ``full-auto`` should be able to see what went through unattended.
    """

    config: AgentConfig
    prompter: Prompter | None = None
    """``None`` means non-interactive: anything needing consent is refused."""

    remembered: dict[str, bool] = field(default_factory=dict)
    asked: int = 0
    auto_approved: int = 0
    aborted: bool = False

    def decide(self, request: ApprovalRequest | None) -> Decision:
        """Resolve one request.

        Args:
            request: What the tool wants permission for, or ``None`` when the
                tool declared the call needs none.

        Returns:
            The decision to act on. ``abort`` set means the user asked to stop.
        """
        if request is None:
            self.auto_approved += 1
            return Decision(approved=True)

        if self._auto_allows(request.risk):
            self.auto_approved += 1
            return Decision(approved=True)

        key = request.signature or request.tool
        if request.risk is not RiskLevel.DANGEROUS and key in self.remembered:
            return Decision(approved=self.remembered[key])

        if self.prompter is None:
            # Non-interactive and consent is required: refusing is the only safe
            # answer, and the model is told so as a tool error it can work around.
            return Decision(approved=False)

        self.asked += 1
        decision = self.prompter(request)
        if decision.abort:
            self.aborted = True
        if decision.remember and request.risk is not RiskLevel.DANGEROUS:
            self.remembered[key] = decision.approved
        return decision

    def requires_prompt(self, request: ApprovalRequest | None) -> bool:
        """Whether :meth:`decide` would actually stop and ask the user.

        Exposed so a UI can announce the question before it is asked, without
        having to reimplement — and eventually contradict — the rules above.
        """
        if request is None or self.prompter is None:
            return False
        if self._auto_allows(request.risk):
            return False
        if request.risk is RiskLevel.DANGEROUS:
            return True
        return (request.signature or request.tool) not in self.remembered

    def _auto_allows(self, risk: RiskLevel) -> bool:
        """Whether this mode runs ``risk`` without asking."""
        if risk is RiskLevel.SAFE:
            return True
        if risk is RiskLevel.DANGEROUS:
            return False
        return self.config.approval_mode == "full-auto"

    def allows_unattended_edits(self) -> bool:
        """Whether file edits skip the prompt in the current mode.

        Consulted by the engine rather than by the tools: an edit is judged by
        where it writes, which is the engine's business, not the file layer's.
        """
        return self.config.approval_mode in ("auto-edit", "full-auto")

    def describe(self) -> str:
        """One line on how this run was supervised, for the closing summary."""
        parts = [f"mode {self.config.approval_mode}"]
        if self.asked:
            parts.append(f"{self.asked} prompt(s)")
        if self.auto_approved:
            parts.append(f"{self.auto_approved} auto-approved")
        if self.remembered:
            parts.append(f"{len(self.remembered)} remembered")
        return ", ".join(parts)
