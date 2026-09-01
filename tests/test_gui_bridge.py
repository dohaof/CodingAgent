"""Desktop bridge protocol regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from cagent.agent.approval import ApprovalPolicy, Decision
from cagent.agent.trace import history_from_trace, read_trace
from cagent.cli.resume import find_trace_choices, trace_step_count
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


def test_trace_step_count_prefers_the_runs_own_closing_count() -> None:
    """``run_finished`` knows the real total; step records can have been clipped."""
    records: list[dict[str, object]] = [
        {"type": "step_finished"},
        {"type": "step_finished"},
        {"type": "run_finished", "steps": 9},
    ]

    assert trace_step_count(records) == 9


def test_trace_step_count_falls_back_to_counting_an_unfinished_run() -> None:
    """An interrupted session never writes ``run_finished``, but it did work."""
    records: list[dict[str, object]] = [
        {"type": "user", "text": "go"},
        {"type": "step_finished"},
        {"type": "step_finished"},
    ]

    assert trace_step_count(records) == 2


class _RecordingEmitter(JsonEmitter):
    """Collects emitted events instead of writing them to stdout."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, object]]] = []

    def send(self, kind: str, **payload: object) -> None:
        self.events.append((kind, payload))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


def _deletable_session(tmp_path: Path) -> tuple[DesktopSession, _RecordingEmitter, Path]:
    """A session whose config points its trace directory at ``tmp_path/traces``."""
    traces = tmp_path / "traces"
    traces.mkdir()
    emitter = _RecordingEmitter()
    session = DesktopSession(tmp_path, emitter)
    session.config = AgentConfig(workspace=tmp_path, trace_dir=traces)
    return session, emitter, traces


def test_deleting_a_session_removes_the_trace_and_refreshes_the_list(tmp_path: Path) -> None:
    session, emitter, traces = _deletable_session(tmp_path)
    victim = traces / "aaaaaaaaaaaa.jsonl"
    _write_trace(victim, "old question", "old answer")

    session.delete_session(victim)

    assert not victim.exists()
    # The sidebar is rebuilt from a `sessions` event, so one must follow the
    # deletion or the row stays on screen until the next turn ends.
    assert emitter.kinds() == ["session_deleted", "sessions"]


def test_deleting_a_path_outside_the_trace_directory_is_refused(tmp_path: Path) -> None:
    """The renderer supplies this path; it must not reach an arbitrary unlink."""
    session, emitter, _ = _deletable_session(tmp_path)
    outsider = tmp_path / "important.jsonl"
    outsider.write_text("keep me", encoding="utf-8")

    session.delete_session(outsider)

    assert outsider.exists()
    assert emitter.kinds() == ["protocol_error"]


def test_deleting_a_non_trace_file_inside_the_trace_directory_is_refused(tmp_path: Path) -> None:
    session, emitter, traces = _deletable_session(tmp_path)
    notes = traces / "notes.txt"
    notes.write_text("keep me", encoding="utf-8")

    session.delete_session(notes)

    assert notes.exists()
    assert emitter.kinds() == ["protocol_error"]


def test_a_resumed_sessions_status_counts_the_steps_it_restored(tmp_path: Path) -> None:
    """The status bar describes the conversation, not the freshly rotated agent.

    Resuming builds a new ``Agent``, so its ``LoopGuard`` restarts at zero. That
    is right for the step *budget* and wrong for the step *count* on screen: the
    footer read "0 steps" beside a sidebar card reporting the same conversation's
    real total. Both numbers now come from :func:`trace_step_count`.
    """
    session, emitter, traces = _deletable_session(tmp_path)
    saved = traces / "dddddddddddd.jsonl"
    _write_trace(saved, "earlier question", "earlier answer")
    session.agent = cast(
        Any,
        SimpleNamespace(
            session_id="rotated-in",
            usage=SimpleNamespace(total=0),
            guard=SimpleNamespace(steps=2),
            context=SimpleNamespace(token_count=lambda: 10, history=[], compactions=0),
            registry=SimpleNamespace(names=lambda: []),
            sandbox_status=lambda: "host",
        ),
    )
    restored = trace_step_count(read_trace(saved))
    session._restored_steps = restored

    session.send_status()

    kind, payload = emitter.events[0]
    assert kind == "status"
    assert payload["steps"] == restored + 2
    # The sidebar card for that same trace must report the part we carried over.
    choice = next(c for c in find_trace_choices(traces) if c.path == saved)
    assert choice.steps == restored


def test_the_live_sessions_own_trace_cannot_be_deleted(tmp_path: Path) -> None:
    """Its writer still holds the file open, so unlinking it loses this run."""
    session, emitter, traces = _deletable_session(tmp_path)
    live = traces / "cccccccccccc.jsonl"
    _write_trace(live, "current question", "current answer")
    session.trace = cast(Any, SimpleNamespace(path=live))

    session.delete_session(live)

    assert live.exists()
    assert emitter.kinds() == ["protocol_error"]
