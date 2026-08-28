"""The agent layer: the loop, its context, its limits, and its reporting.

Import from the submodules rather than relying on names re-exported here, which
are only the ones a host application needs to start a session.
"""

from __future__ import annotations

from .approval import ApprovalPolicy, Decision
from .context import ContextManager
from .engine import Agent, TurnResult
from .events import AgentEvent, CollectingSink, EventSink, FanOutSink
from .guards import LoopGuard
from .prompt import PromptBuilder
from .sandbox import SandboxError, SandboxSession
from .trace import TraceWriter, history_from_trace, read_trace

__all__ = [
    "Agent",
    "AgentEvent",
    "ApprovalPolicy",
    "CollectingSink",
    "ContextManager",
    "Decision",
    "EventSink",
    "FanOutSink",
    "LoopGuard",
    "PromptBuilder",
    "SandboxError",
    "SandboxSession",
    "TraceWriter",
    "TurnResult",
    "history_from_trace",
    "read_trace",
]
