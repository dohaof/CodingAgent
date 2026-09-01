"""Desktop bridge protocol regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from cagent.agent.approval import ApprovalPolicy, Decision
from cagent.agent.trace import history_from_trace, read_trace
from cagent.config import AgentConfig
from cagent.gui.bridge import DesktopSession, JsonEmitter, _jsonable
from cagent.types import Message, RiskLevel, TextPart, ThinkingPart, ToolCallPart


def test_bridge_serializes_int_enum_as_a_risk_name() -> None:
    assert _jsonable(RiskLevel.DANGEROUS) == "dangerous"


def test_bridge_adds_discriminators_to_message_parts() -> None:
    message = Message.assistant(
        ThinkingPart("checking"),
        TextPart("done"),
        ToolCallPart("call-1", "read_file", {"path": "app.py"}),
    )

    encoded = _jsonable(message)

    assert isinstance(encoded, dict)
    assert [part["type"] for part in encoded["parts"]] == [
        "thinking",
        "text",
        "tool_call",
    ]


def _write_trace(path: Path, prompt: str, answer: str) -> None:
    records = [
        {"type": "session", "session_id": path.stem, "workspace": str(path.parent)},
        {"type": "user", "text": prompt},
        {
            "type": "step_finished",
            "message": {"role": "assistant", "parts": [{"type": "text", "text": answer}]},
        },
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_a_resumed_trace_is_not_overwritten_by_a_later_checkpoint(tmp_path: Path) -> None:
    """A checkpoint written into a trace must not replace that trace's own turns.

    The desktop client can resume while a session already has history. Writing
    the resumed history into the live trace makes every later replay of it
    return the grafted conversation instead of the one it recorded.
    """
    first = tmp_path / "aaaaaaaaaaaa.jsonl"
    second = tmp_path / "bbbbbbbbbbbb.jsonl"
    _write_trace(first, "question one", "answer one")
    _write_trace(second, "question two", "answer two")

    # Simulate the old bridge behaviour: resume `first`, then persist that
    # history as a checkpoint on `second`, the session that was still live.
    grafted = history_from_trace(read_trace(first))
    with second.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "history_checkpoint", "messages": [_jsonable(m) for m in grafted]},
                ensure_ascii=False,
            )
            + "\n"
        )

    replayed = [m.text for m in history_from_trace(read_trace(second))]

    assert replayed == ["question one", "answer one"]
    assert [m.text for m in history_from_trace(read_trace(first))] == replayed


class _StubAgent:
    """Stands in for the agent: records the teardown calls close() makes."""

    def __init__(self) -> None:
        self.finished: list[str] = []
        self.closed = False

    def interrupt(self) -> None:
        pass

    def finish(self, reason: str, *, trace_path: str | None = None) -> None:
        self.finished.append(reason)

    def close(self) -> None:
        self.closed = True


def _idle_session(tmp_path: Path) -> DesktopSession:
    """A constructed-but-never-started session with a policy that would answer."""
    session = DesktopSession(tmp_path, JsonEmitter())
    session.policy = ApprovalPolicy(config=AgentConfig(), prompter=lambda _request: Decision(True))
    session.agent = cast(Any, _StubAgent())
    return session


def test_closing_a_session_under_a_live_window_keeps_the_approval_prompter(tmp_path: Path) -> None:
    """Rotating a session must not silently discard sandbox changes.

    ``Agent._finish_sandbox`` asks for approval before copying a disposable
    workspace back to the project. With no prompter the policy refuses, and the
    refusal path throws the changes away — so clearing the prompter is only
    correct when there is genuinely nobody left to ask. Starting a new session
    happens under an open window, where the question can and must be asked.
    """
    session = _idle_session(tmp_path)

    session.close("new_session", interactive=True)

    assert session.policy is not None
    assert session.policy.prompter is not None
    assert cast(Any, session.agent).finished == ["new_session"]


def test_closing_a_session_with_no_renderer_drops_the_approval_prompter(tmp_path: Path) -> None:
    """The other half of the rule: never block process exit on a dead window."""
    session = _idle_session(tmp_path)

    session.close("bridge_closed")

    assert session.policy is not None
    assert session.policy.prompter is None
    assert cast(Any, session.agent).closed is True
