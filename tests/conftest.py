"""Shared fixtures.

Everything here exists to keep the tests hermetic: no network, no real sleeps,
no dependence on the machine's PATH or on files outside ``tmp_path``. Where a
test needs a model, it gets a scripted one whose exact requests it can inspect.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cagent.agent import sandbox as sandbox_module
from cagent.agent.approval import ApprovalPolicy, Decision
from cagent.agent.events import CollectingSink
from cagent.config import AgentConfig
from cagent.llm.base import (
    LLMProvider,
    StreamEvent,
    StreamFinished,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStarted,
    UsageReport,
)
from cagent.tools.base import ApprovalRequest, ToolContext
from cagent.types import Message, ToolSpec, Usage


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch, tmp_path_factory) -> None:
    """Keep the developer's own configuration out of every test.

    ``load_config`` reads ``~/.cagent.toml`` and ``./.cagent.toml`` by design,
    which means a real file on the machine running the tests would silently
    supply an endpoint, a model, or a key — and a test asserting that one is
    *missing* would pass or fail depending on whose laptop it ran on. Pointing
    home and the working directory at an empty directory removes that.

    The directory is deliberately outside the test's own ``tmp_path``, which
    many tests list or assert the contents of.
    """
    empty = tmp_path_factory.mktemp("no_config")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty))
    monkeypatch.chdir(empty)


@pytest.fixture(autouse=True)
def assume_local_sandbox_image(monkeypatch) -> None:
    """Keep sandbox image lookups off the machine's real Docker daemon.

    ``create_with_status`` verifies the configured image before it snapshots, in
    both ``auto`` and ``docker`` mode, so a test that enables a sandbox would
    otherwise pass or fail depending on which images the developer happens to
    have pulled. Tests about a *missing* image override this with their own
    ``setattr``.
    """
    monkeypatch.setattr(sandbox_module, "docker_image_available", lambda _image: True)


@pytest.fixture
def config(tmp_path: Path) -> AgentConfig:
    """A fully specified config rooted in a temporary workspace.

    An endpoint is four settings and none of them have defaults, so every test
    that talks to a provider states all of them.
    """
    return AgentConfig(
        workspace=tmp_path,
        base_url="https://api.test.invalid/v1",
        model="test-model",
        api_key="test-key",
        sandbox_mode="off",
    )


@dataclass
class ContextHarness:
    """A :class:`ToolContext` plus what the tools did with it."""

    ctx: ToolContext
    emitted: list[str] = field(default_factory=list)
    requests: list[ApprovalRequest] = field(default_factory=list)


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., ContextHarness]:
    """Factory for tool contexts.

    ``approve`` defaults to allowing everything, since most tool tests are about
    the tool's own behaviour; pass ``approve=False`` to test a refusal.
    """

    def factory(
        *,
        approve: bool = True,
        workspace: Path | None = None,
        **config_kwargs: object,
    ) -> ContextHarness:
        root = workspace or tmp_path
        # Tool unit tests use the host explicitly. Automatic Docker selection
        # has dedicated lifecycle tests and must not depend on the test host.
        config_kwargs.setdefault("sandbox_mode", "off")
        cfg = AgentConfig(workspace=root, api_key="test-key", **config_kwargs)  # type: ignore[arg-type]
        harness = ContextHarness(ctx=None)  # type: ignore[arg-type]

        def record(request: ApprovalRequest) -> bool:
            harness.requests.append(request)
            return approve

        harness.ctx = ToolContext(
            workspace=root,
            config=cfg,
            approve=record,
            emit=harness.emitted.append,
            abort=threading.Event(),
        )
        return harness

    return factory


class ScriptedProvider(LLMProvider):
    """A provider that replays scripted turns and records every request.

    Deliberately not a ``Mock``: the assertions worth making are about the
    request bodies the engine produced, so the double keeps them.
    """

    wire = "scripted"

    def __init__(self, config: AgentConfig, script: Sequence[Sequence[StreamEvent]]) -> None:
        self.config = config
        self.script = [list(turn) for turn in script]
        self.requests: list[list[Message]] = []
        self.systems: list[str] = []
        self.tool_specs: list[tuple[ToolSpec, ...]] = []
        self._owns_client = False
        self._client = None  # type: ignore[assignment]

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec] = (),
        abort: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        # Snapshot, so later compaction cannot retroactively change what a test
        # believes was sent. ``synthetic`` rides along: several assertions turn
        # on telling the user's own turns from engine-inserted context.
        self.requests.append(
            [Message(m.role, list(m.parts), synthetic=m.synthetic) for m in messages]
        )
        self.systems.append(system)
        self.tool_specs.append(tuple(tools))
        turn = self.script.pop(0) if self.script else [TextDelta("done"), StreamFinished("stop")]
        yield from turn

    def close(self) -> None:
        return None

    @property
    def last_tool_results(self) -> list[str]:
        """Contents of the tool results in the most recent request."""
        if not self.requests:
            return []
        return [
            part.content
            for message in self.requests[-1]
            if message.role == "tool"
            for part in message.tool_results
        ]

    def pairing_is_valid(self) -> bool:
        """Whether every request paired each tool call with exactly one result.

        The invariant both wire formats enforce, checked across the whole run
        rather than at one point, because compaction is what tends to break it.
        """
        for messages in self.requests:
            calls = [part.id for m in messages for part in m.tool_calls]
            results = [part.call_id for m in messages for part in m.tool_results]
            if calls != results:
                return False
        return True


def tool_turn(
    name: str,
    arguments: dict[str, object],
    *,
    index: int = 0,
    split: int = 2,
) -> list[StreamEvent]:
    """Events for one tool call, arguments split across ``split`` fragments.

    Splitting by default is the point: providers stream argument JSON in pieces,
    and a test that sends it whole would not exercise accumulation.
    """
    blob = json.dumps(arguments)
    size = max(len(blob) // max(split, 1), 1)
    fragments = [blob[i : i + size] for i in range(0, len(blob), size)] or [""]
    events: list[StreamEvent] = [ToolCallStarted(index=index, id=f"call_{index}", name=name)]
    events += [ToolCallArgsDelta(index=index, delta=piece) for piece in fragments]
    return events


def text_turn(text: str, *, usage: Usage | None = None) -> list[StreamEvent]:
    """Events for a prose answer that ends the turn."""
    events: list[StreamEvent] = [TextDelta(text)]
    if usage is not None:
        events.append(UsageReport(usage))
    events.append(StreamFinished("stop"))
    return events


@pytest.fixture
def scripted() -> Callable[[AgentConfig, Sequence[Sequence[StreamEvent]]], ScriptedProvider]:
    """Factory for :class:`ScriptedProvider`."""

    def factory(
        config: AgentConfig, script: Sequence[Sequence[StreamEvent]]
    ) -> ScriptedProvider:
        return ScriptedProvider(config, script)

    return factory


@pytest.fixture
def sink() -> CollectingSink:
    """An event sink that records everything for later assertions."""
    return CollectingSink()


@pytest.fixture
def allow_all(config: AgentConfig) -> ApprovalPolicy:
    """A policy that approves whatever it is asked."""
    return ApprovalPolicy(config, prompter=lambda request: Decision(approved=True))


@pytest.fixture
def deny_all(config: AgentConfig) -> ApprovalPolicy:
    """A policy that refuses whatever it is asked."""
    return ApprovalPolicy(config, prompter=lambda request: Decision(approved=False))


def sse_body(chunks: Sequence[dict[str, object]], *, done: bool = True) -> bytes:
    """Assemble an OpenAI-style SSE response body."""
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def anthropic_sse(events: Sequence[tuple[str, dict[str, object]]]) -> bytes:
    """Assemble an Anthropic-style SSE response body, with event names."""
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    ).encode()


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    """A sleep that records instead of waiting, for retry tests."""
    calls: list[float] = []

    def fake(delay: float) -> None:
        calls.append(delay)

    fake.calls = calls  # type: ignore[attr-defined]
    return fake
