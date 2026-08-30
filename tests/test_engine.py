"""The agent loop, driven by a scripted model.

These are the tests that describe what the agent *is*: read, act, observe,
repeat, stop. They run with no network and no real model, which is what makes it
possible to assert on the awkward paths — a refused edit, a malformed tool call,
a model that loops — that a live run produces only by accident.

The recurring assertion is that a failure becomes a tool result rather than an
exception, because that is the mechanism by which the agent corrects itself.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cagent.agent import sandbox as sandbox_module
from cagent.agent.approval import ApprovalPolicy, Decision
from cagent.agent.engine import Agent
from cagent.agent.events import (
    Activity,
    ApprovalRequested,
    CollectingSink,
    CompactionDone,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    UserMessage,
    Warning,
)
from cagent.config import AgentConfig
from cagent.llm.base import (
    StreamFinished,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallStarted,
    UsageReport,
)
from cagent.llm.base import TextDelta as WireTextDelta
from cagent.tools.base import ApprovalRequest, ToolOutcome
from cagent.tools.registry import ToolRegistry, default_registry, tool
from cagent.types import (
    FinishReason,
    Message,
    RiskLevel,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from tests.conftest import ScriptedProvider, text_turn, tool_turn


@tool(risk=RiskLevel.SAFE)
def explode() -> ToolOutcome:
    """Always fail."""
    raise RuntimeError("internal boom")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A workspace holding a real bug that its own tests detect."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    return tmp_path


def make_agent(
    config: AgentConfig,
    script: list[list[object]],
    *,
    policy: ApprovalPolicy | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[Agent, CollectingSink, ScriptedProvider]:
    sink = CollectingSink()
    provider = ScriptedProvider(config, script)  # type: ignore[arg-type]
    agent = Agent(
        config=config,
        provider=provider,
        registry=registry or default_registry(),
        policy=policy or ApprovalPolicy(config),
        sink=sink,
    )
    return agent, sink, provider


def auto(project: Path, **kwargs: object) -> AgentConfig:
    """A config that runs unattended, for tests not about approval."""
    return AgentConfig(
        workspace=project,
        api_key="k",
        base_url="https://api.test.invalid/v1",
        model="test-model",
        approval_mode="full-auto",
        allow_outside_workspace=bool(kwargs.pop("allow_outside_workspace", False)),
        sandbox_mode=kwargs.pop("sandbox_mode", "off"),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheCycle:
    def test_initial_prompt_can_be_deferred_for_an_interactive_frontend(
        self, project: Path
    ) -> None:
        config = auto(project)
        sink = CollectingSink()
        agent = Agent(
            config=config,
            provider=ScriptedProvider(config, []),
            registry=default_registry(),
            policy=ApprovalPolicy(config),
            sink=sink,
            defer_initial_prompt=True,
        )

        assert agent.context.system_tokens == 0
        assert agent.prompt_builder._index is None

        agent.initialize()

        assert agent.context.system_tokens > 0
        assert agent.prompt_builder._index is not None

    def test_undo_removes_a_whole_user_turn_including_tool_pairs(
        self, project: Path
    ) -> None:
        agent, _, _ = make_agent(auto(project), [])
        calls = [
            ToolCallPart("a", "read_file", {"path": "a.py"}),
            ToolCallPart("b", "read_file", {"path": "b.py"}),
        ]
        agent.context.history = [
            Message.user("first request"),
            Message.assistant(TextPart("first answer")),
            Message.user("second request"),
            Message(
                role="user",
                parts=[TextPart("internal compaction note")],
                synthetic=True,
            ),
            Message.assistant(*calls),
            Message.from_tool_results(
                [ToolResultPart("a", "A"), ToolResultPart("b", "B")]
            ),
            Message.assistant(TextPart("second answer")),
        ]

        removed = agent.undo_last_turn()

        assert removed == 5
        assert [message.text for message in agent.context.history] == [
            "first request",
            "first answer",
        ]
        assert agent.undo_last_turn() == 2
        assert agent.context.history == []
        assert agent.undo_last_turn() == 0

    def test_restored_history_is_sent_before_the_next_turn(self, project: Path) -> None:
        agent, _, provider = make_agent(auto(project), [text_turn("continued")])
        agent.restore_history(
            [Message.user("old task"), Message.assistant(TextPart("old answer"))]
        )

        result = agent.run_turn("continue the work")

        assert result.completed
        assert [message.role for message in provider.requests[0]] == [
            "user",
            "assistant",
            "user",
        ]
        assert provider.requests[0][0].text == "old task"
        assert provider.requests[0][1].text == "old answer"

    def test_a_prose_answer_ends_the_turn_in_one_step(self, project: Path) -> None:
        agent, sink, provider = make_agent(auto(project), [text_turn("Nothing to do.")])
        result = agent.run_turn("say hello")
        assert result.completed and result.steps == 1
        assert result.reply == "Nothing to do."
        assert len(provider.requests) == 1

    def test_read_edit_verify_answer(self, project: Path) -> None:
        # The whole point of the project, end to end: the agent inspects, changes,
        # runs the real test suite, and reports.
        agent, sink, provider = make_agent(
            auto(project, allow_outside_workspace=True),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                tool_turn("run_bash", {"command": "python -m pytest -q"})
                + [StreamFinished("tool_calls")],
                tool_turn(
                    "edit_file",
                    {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"},
                )
                + [StreamFinished("tool_calls")],
                tool_turn("run_bash", {"command": "python -m pytest -q"})
                + [StreamFinished("tool_calls")],
                text_turn("Fixed the operator in calc.py:2; pytest passes."),
            ],
        )

        result = agent.run_turn("fix the failing test")

        assert result.completed, result
        fixed = (project / "calc.py").read_text(encoding="utf-8")
        assert fixed == "def add(a, b):\n    return a + b\n"
        assert [e.call.name for e in sink.of_type(ToolStarted)] == [
            "read_file",
            "run_bash",
            "edit_file",
            "run_bash",
        ]
        runs = [e for e in sink.of_type(ToolFinished) if e.call.name == "run_bash"]
        assert runs[0].outcome.is_error and not runs[1].outcome.is_error
        assert provider.pairing_is_valid()

    def test_a_failing_command_is_handed_back_with_its_stderr(self, project: Path) -> None:
        # Without the traceback reaching the transcript there is no self-correction.
        agent, sink, provider = make_agent(
            auto(project, allow_outside_workspace=True),
            [
                tool_turn("run_bash", {"command": "python -m pytest -q"})
                + [StreamFinished("tool_calls")],
                text_turn("The test fails because add subtracts."),
            ],
        )
        agent.run_turn("run the tests")
        (fed,) = provider.last_tool_results
        assert "assert" in fed.lower() and "exit code: 1" in fed

    def test_explicit_read_only_calls_run_concurrently_in_result_order(
        self, project: Path
    ) -> None:
        lock = threading.Lock()
        both_running = threading.Event()
        active = 0
        peak = 0

        @tool(risk=RiskLevel.SAFE, parallel_safe=True)
        def probe(value: str) -> str:
            """Observe concurrent execution."""
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_running.set()
            both_running.wait(timeout=2)
            with lock:
                active -= 1
            return value

        registry = ToolRegistry()
        registry.register_class(probe)
        turn = (
            tool_turn("probe", {"value": "first"}, index=0)
            + tool_turn("probe", {"value": "second"}, index=1)
            + [StreamFinished("tool_calls")]
        )
        agent, sink, provider = make_agent(
            auto(project), [turn, text_turn("done")], registry=registry
        )

        agent.run_turn("probe twice")

        assert peak == 2
        assert len(sink.of_type(ToolFinished)) == 2
        assert provider.last_tool_results == ["first", "second"]
        assert provider.pairing_is_valid()

    def test_default_parallel_tools_are_explicitly_read_only(self) -> None:
        registry = default_registry()
        parallel = {tool.name for tool in registry if tool.parallel_safe}

        assert parallel == {"read_file", "list_dir", "glob_files", "grep_search"}

    def test_a_batch_mixing_reads_and_writes_remains_serial(self, project: Path) -> None:
        turn = (
            tool_turn("read_file", {"path": "calc.py"}, index=0)
            + tool_turn("write_file", {"path": "note.txt", "content": "done\n"}, index=1)
            + [StreamFinished("tool_calls")]
        )
        agent, sink, provider = make_agent(auto(project), [turn, text_turn("done")])

        agent.run_turn("read and write")

        lifecycle = [
            (type(event).__name__, event.call.name)
            for event in sink.events
            if isinstance(event, ToolStarted | ToolFinished)
        ]
        assert lifecycle == [
            ("ToolStarted", "read_file"),
            ("ToolFinished", "read_file"),
            ("ToolStarted", "write_file"),
            ("ToolFinished", "write_file"),
        ]
        assert provider.pairing_is_valid()

    def test_usage_accumulates_across_steps(self, project: Path) -> None:
        agent, _, _ = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"})
                + [UsageReport(Usage(100, 20)), StreamFinished("tool_calls")],
                text_turn("done", usage=Usage(150, 30)),
            ],
        )
        result = agent.run_turn("read it")
        assert agent.usage.prompt_tokens == 250
        assert agent.usage.completion_tokens == 50
        assert result.usage.prompt_tokens == 250

    def test_the_system_prompt_and_tools_go_with_every_request(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
        )
        agent.run_turn("read it")
        assert all(system for system in provider.systems)
        assert all("read_file" in {s.name for s in specs} for specs in provider.tool_specs)

    def test_the_current_task_ranks_the_repo_map_sent_to_the_provider(
        self, project: Path
    ) -> None:
        target = project / "deep" / "transport"
        target.mkdir(parents=True)
        (target / "stream_controller.py").write_text(
            "def cancel_provider_stream(): pass\n", encoding="utf-8"
        )
        agent, _, provider = make_agent(auto(project), [text_turn("done")])

        agent.run_turn("repair provider stream cancellation")

        system = provider.systems[0]
        assert system.index("deep/transport/stream_controller.py") < system.index("calc.py")

    def test_the_transcript_grows_by_assistant_and_tool_turns(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
        )
        agent.run_turn("read it")
        roles = [m.role for m in agent.context.history]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_events_narrate_the_run_in_order(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
        )
        agent.announce("read it")
        agent.run_turn("read it")
        agent.finish("finished")
        kinds = [type(event).__name__ for event in sink.events]
        assert kinds[0] == "RunStarted"
        assert kinds.index("UserMessage") < kinds.index("StepStarted")
        assert kinds.index("ToolStarted") < kinds.index("ToolFinished")
        assert kinds[-1] == "RunFinished"

    def test_activity_events_identify_task_context_and_model_wait(self, project: Path) -> None:
        agent, sink, _ = make_agent(auto(project), [text_turn("done")])

        agent.run_turn("read it")

        assert [event.message for event in sink.of_type(Activity)] == [
            "Preparing task context",
            "Waiting for agent response",
        ]
        kinds = [type(event).__name__ for event in sink.events]
        assert kinds.index("Activity") < kinds.index("UserMessage")
        assert kinds.index("UserMessage") < kinds.index("StepStarted")

    def test_activity_reports_context_preparation_when_repo_map_is_disabled(
        self, project: Path
    ) -> None:
        config = auto(project, repo_map_enabled=False)
        agent, sink, _ = make_agent(config, [text_turn("done")])

        agent.run_turn("read it")

        assert [event.message for event in sink.of_type(Activity)] == [
            "Preparing context",
            "Waiting for agent response",
        ]

    def test_streamed_text_is_republished_for_the_ui(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project),
            [[WireTextDelta("Hel"), WireTextDelta("lo"), StreamFinished("stop")]],
        )
        agent.run_turn("greet")
        assert "".join(e.text for e in sink.of_type(TextDelta)) == "Hello"

    def test_thinking_is_republished_separately(self, project: Path) -> None:
        from cagent.agent.events import ThinkingDelta as EventThinkingDelta

        agent, sink, _ = make_agent(
            auto(project),
            [[ThinkingDelta("considering"), WireTextDelta("answer"), StreamFinished("stop")]],
        )
        agent.run_turn("think")
        assert [e.text for e in sink.of_type(EventThinkingDelta)] == ["considering"]

    def test_a_second_turn_continues_the_same_transcript(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project), [text_turn("first"), text_turn("second")]
        )
        agent.run_turn("one")
        agent.run_turn("two")
        assert len(provider.requests[1]) > len(provider.requests[0])
        assert provider.requests[1][0].text == "one"


class TestFailureHandling:
    def test_auto_sandbox_uses_a_local_image_when_docker_is_ready(
        self, project: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
        monkeypatch.setattr(sandbox_module, "docker_image_available", lambda _image: True)
        config = auto(
            project,
            sandbox_mode="auto",
            sandbox_sync="never",
        )
        agent, sink, _ = make_agent(config, [text_turn("done")])
        try:
            assert agent.sandbox is not None
            agent.announce("test isolation")
            (started,) = sink.of_type(RunStarted)
            assert started.shell_access == "container"
            assert "docker" in started.sandbox_status
            assert not sink.of_type(Warning)
        finally:
            agent.close()

    def test_auto_sandbox_warns_before_falling_back_to_the_host(
        self, project: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sandbox_module, "docker_available", lambda: False)
        config = auto(project, sandbox_mode="auto")
        agent, sink, _ = make_agent(config, [text_turn("done")])
        try:
            assert agent.sandbox is None
            agent.announce("test fallback")
            (started,) = sink.of_type(RunStarted)
            assert started.path_boundary == "workspace-only"
            assert started.shell_access == "host (unrestricted)"
            (warning,) = sink.of_type(Warning)
            assert "unrestricted process access" in warning.message
            assert warning.detail is not None and "daemon is unavailable" in warning.detail
        finally:
            agent.close()

    def test_sandbox_changes_can_be_applied_or_discarded_mid_session(
        self, project: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
        config = auto(project, sandbox_sync="ask")
        agent, _, _ = make_agent(config, [text_turn("done")])
        try:
            agent.enable_sandbox()
            assert agent.sandbox is not None
            (agent.sandbox.workspace / "applied.txt").write_text("keep\n", encoding="utf-8")
            assert agent.apply_sandbox_changes() == ("applied.txt",)
            assert (project / "applied.txt").read_text(encoding="utf-8") == "keep\n"

            (agent.sandbox.workspace / "discarded.txt").write_text("drop\n", encoding="utf-8")
            agent.discard_sandbox_changes()
            assert not (project / "discarded.txt").exists()
            assert agent.sandbox is not None
        finally:
            agent.close()

    def test_sandbox_can_be_toggled_between_turns(self, project: Path, monkeypatch) -> None:
        monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
        config = auto(project, sandbox_sync="always")
        agent, _, _ = make_agent(config, [text_turn("done")])
        try:
            assert agent.sandbox_status() == "off"
            agent.enable_sandbox(image="ubuntu:22.04")
            assert agent.sandbox is not None
            assert "docker" in agent.sandbox_status()
            (agent.sandbox.workspace / "from-sandbox.txt").write_text("ok\n", encoding="utf-8")

            agent.disable_sandbox()
            assert agent.sandbox is None
            assert agent.sandbox_status() == "off"
            assert (project / "from-sandbox.txt").read_text(encoding="utf-8") == "ok\n"
        finally:
            agent.close()

    def test_sandbox_edits_are_synced_only_after_final_approval(
        self, project: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
        config = AgentConfig(
            workspace=project,
            api_key="k",
            base_url="https://api.test.invalid/v1",
            model="test-model",
            approval_mode="auto-edit",
            sandbox_mode="docker",
            sandbox_sync="ask",
        )
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda request: (asked.append(request), Decision(True))[1]
        )
        agent, sink, _ = make_agent(
            config,
            [
                tool_turn(
                    "edit_file",
                    {
                        "path": "calc.py",
                        "old_string": "return a - b",
                        "new_string": "return a + b",
                    },
                )
                + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
            policy=policy,
        )

        agent.run_turn("fix it in the sandbox")
        assert "return a - b" in (project / "calc.py").read_text(encoding="utf-8")

        agent.finish("finished")
        assert "return a + b" in (project / "calc.py").read_text(encoding="utf-8")
        assert [request.tool for request in asked] == ["sandbox_sync"]
        assert asked[0].always_prompt
        assert "+    return a + b" in (asked[0].detail or "")
        assert sink.of_type(RunFinished)
        agent.close()

    def test_auto_edit_executes_edits_but_still_asks_for_shell_writes(
        self, project: Path
    ) -> None:
        config = AgentConfig(
            workspace=project,
            api_key="k",
            base_url="https://api.test.invalid/v1",
            model="test-model",
            approval_mode="auto-edit",
            allow_outside_workspace=True,
        )
        asked: list[str] = []

        def refuse(request: ApprovalRequest) -> Decision:
            asked.append(request.tool)
            return Decision(approved=False)

        agent, sink, provider = make_agent(
            config,
            [
                tool_turn(
                    "edit_file",
                    {
                        "path": "calc.py",
                        "old_string": "return a - b",
                        "new_string": "return a + b",
                    },
                )
                + [StreamFinished("tool_calls")],
                tool_turn("run_bash", {"command": "echo changed > marker.txt"})
                + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
            policy=ApprovalPolicy(config, prompter=refuse),
        )

        result = agent.run_turn("fix the code and write a marker")

        assert result.completed
        assert "return a + b" in (project / "calc.py").read_text(encoding="utf-8")
        assert not (project / "marker.txt").exists()
        assert asked == ["run_bash"]
        (prompt,) = sink.of_type(ApprovalRequested)
        assert prompt.request.tool == "run_bash"
        assert prompt.request.risk is RiskLevel.MUTATING
        assert any("declined" in result for result in provider.last_tool_results)

    def test_a_refused_call_is_reported_to_the_model_and_changes_nothing(
        self, project: Path
    ) -> None:
        config = AgentConfig(
            workspace=project, api_key="k", base_url="https://api.test.invalid/v1",
            model="test-model", approval_mode="suggest", allow_outside_workspace=True
        )
        policy = ApprovalPolicy(config, prompter=lambda request: Decision(approved=False))
        before = (project / "calc.py").read_bytes()

        agent, sink, provider = make_agent(
            config,
            [
                tool_turn(
                    "edit_file",
                    {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"},
                )
                + [StreamFinished("tool_calls")],
                text_turn("I need permission for that."),
            ],
            policy=policy,
        )
        result = agent.run_turn("fix it")

        assert result.completed
        assert (project / "calc.py").read_bytes() == before
        (fed,) = provider.last_tool_results
        assert "declined" in fed and "Do not retry" in fed
        assert sink.of_type(ApprovalRequested)

    def test_the_approval_prompt_carries_the_real_diff(self, project: Path) -> None:
        config = AgentConfig(
            workspace=project, api_key="k", base_url="https://api.test.invalid/v1",
            model="test-model", approval_mode="suggest", allow_outside_workspace=True
        )
        policy = ApprovalPolicy(config, prompter=lambda request: Decision(approved=True))
        agent, sink, _ = make_agent(
            config,
            [
                tool_turn(
                    "edit_file",
                    {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"},
                )
                + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
            policy=policy,
        )
        agent.run_turn("fix it")
        (event,) = sink.of_type(ApprovalRequested)
        assert event.request.detail is not None
        assert "+    return a + b" in event.request.detail

    def test_an_abort_from_the_prompt_stops_the_turn(self, project: Path) -> None:
        config = AgentConfig(
            workspace=project, api_key="k", base_url="https://api.test.invalid/v1",
            model="test-model", approval_mode="suggest", allow_outside_workspace=True
        )
        policy = ApprovalPolicy(
            config, prompter=lambda request: Decision(approved=False, abort=True)
        )
        agent, _, _ = make_agent(
            config,
            [
                tool_turn("run_bash", {"command": "pytest"}) + [StreamFinished("tool_calls")],
                text_turn("unreachable"),
            ],
            policy=policy,
        )
        result = agent.run_turn("run the tests")
        assert result.stopped_by == "aborted"

    def test_malformed_tool_arguments_ask_for_valid_json(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project),
            [
                [
                    ToolCallStarted(index=0, id="bad", name="read_file"),
                    ToolCallArgsDelta(index=0, delta='{"path": broken'),
                    StreamFinished("tool_calls"),
                ],
                text_turn("retrying"),
            ],
        )
        agent.run_turn("read it")
        (fed,) = provider.last_tool_results
        assert "valid JSON" in fed
        assert provider.pairing_is_valid()

    def test_an_unknown_tool_is_answered_with_the_real_names(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project),
            [
                tool_turn("read_filez", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("retrying"),
            ],
        )
        agent.run_turn("read it")
        (fed,) = provider.last_tool_results
        assert "read_file" in fed and "read_filez" in fed

    def test_bad_arguments_are_answered_without_running_the_tool(self, project: Path) -> None:
        agent, sink, provider = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"wrong_field": "x"}) + [StreamFinished("tool_calls")],
                text_turn("retrying"),
            ],
        )
        agent.run_turn("read it")
        (fed,) = provider.last_tool_results
        assert "path" in fed and "wrong_field" in fed

    def test_a_tool_that_crashes_does_not_end_the_loop(self, project: Path) -> None:
        registry = ToolRegistry()
        registry.register_class(explode)

        agent, _, provider = make_agent(
            auto(project),
            [
                tool_turn("explode", {}) + [StreamFinished("tool_calls")],
                text_turn("noted the crash"),
            ],
            registry=registry,
        )
        result = agent.run_turn("explode please")
        assert result.completed
        (fed,) = provider.last_tool_results
        assert "RuntimeError" in fed and "internal boom" in fed

    def test_a_provider_failure_is_reported_not_raised(self, project: Path) -> None:
        from cagent.errors import AuthError

        class Failing(ScriptedProvider):
            def stream(self, messages, *, system, tools=(), abort=None):  # type: ignore[no-untyped-def]
                raise AuthError("invalid key")
                yield  # pragma: no cover

        config = auto(project)
        sink = CollectingSink()
        agent = Agent(
            config=config,
            provider=Failing(config, []),
            registry=default_registry(),
            policy=ApprovalPolicy(config),
            sink=sink,
        )
        result = agent.run_turn("do something")
        assert result.stopped_by == "provider_error"
        assert "invalid key" in result.reply
        assert sink.of_type(Warning)

    def test_a_truncated_reply_is_flagged(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project), [[WireTextDelta("half a thought"), StreamFinished("length")]]
        )
        result = agent.run_turn("write an essay")
        assert not result.completed and result.stopped_by == "length"
        assert any("output token limit" in event.message for event in sink.of_type(Warning))

    @pytest.mark.parametrize("reason", ["content_filter", "error"])
    def test_abnormal_model_stops_are_not_reported_as_complete(
        self, project: Path, reason: FinishReason
    ) -> None:
        agent, sink, _ = make_agent(
            auto(project), [[WireTextDelta("partial"), StreamFinished(reason)]]
        )

        result = agent.run_turn("answer carefully")

        assert not result.completed and result.stopped_by == reason
        assert sink.of_type(Warning)

    def test_a_partial_tool_call_does_not_pollute_history(self, project: Path) -> None:
        partial_call = tool_turn("read_file", {"path": "calc.py"}) + [
            StreamFinished("length")
        ]
        agent, _, provider = make_agent(auto(project), [partial_call, text_turn("continued")])

        first = agent.run_turn("inspect it")
        second = agent.run_turn("continue")

        assert first.stopped_by == "length" and second.completed
        assert provider.pairing_is_valid()

    def test_paths_outside_the_workspace_are_refused(self, project: Path) -> None:
        agent, _, provider = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "../../etc/passwd"})
                + [StreamFinished("tool_calls")],
                text_turn("blocked"),
            ],
        )
        agent.run_turn("read the password file")
        (fed,) = provider.last_tool_results
        assert "outside" in fed.lower() or "workspace" in fed.lower()


class TestTermination:
    def test_the_token_budget_stops_the_turn(self, project: Path) -> None:
        script = [
            tool_turn("read_file", {"path": "calc.py", "offset": n})
            + [UsageReport(Usage(500, 100)), StreamFinished("tool_calls")]
            for n in range(1, 10)
        ]
        agent, _, _ = make_agent(auto(project, token_budget=700), script)
        result = agent.run_turn("keep reading")
        assert result.stopped_by == "TokenBudgetExceeded"
        assert "--token-budget" in result.reply

    def test_a_looping_model_is_nudged_then_stopped(self, project: Path) -> None:
        same = tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")]
        agent, _, provider = make_agent(
            auto(project, max_repeated_calls=2), [list(same) for _ in range(8)]
        )
        result = agent.run_turn("read it repeatedly")

        assert result.stopped_by == "RepetitionDetected"
        nudged = [
            content
            for request in provider.requests
            for message in request
            if message.role == "tool"
            for part in message.tool_results
            if "identical arguments" in (content := part.content)
        ]
        assert nudged, "the model was stopped without ever being told it was looping"
        agent.run_turn("continue")
        assert provider.pairing_is_valid()

    def test_an_interrupt_ends_the_turn_at_the_next_step(self, project: Path) -> None:
        agent, _, _ = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("unreachable"),
            ],
        )
        agent.interrupt()
        result = agent.run_turn("read it")
        assert result.stopped_by == "aborted"

    def test_an_interrupt_can_be_cleared_for_the_next_turn(self, project: Path) -> None:
        agent, _, _ = make_agent(auto(project), [text_turn("ignored"), text_turn("second")])
        agent.interrupt()
        assert agent.run_turn("one").stopped_by == "aborted"
        agent.policy.aborted = True
        agent.reset_interrupt()
        assert not agent.policy.aborted
        assert agent.run_turn("two").completed


class TestContextPressure:
    def test_history_is_compacted_and_stays_valid(self, project: Path) -> None:
        (project / "noisy.py").write_text(
            "for i in range(200): print('output line', i, 'x' * 40)\n", encoding="utf-8"
        )
        config = auto(
            project,
            allow_outside_workspace=True,
            context_window=4000,
            compact_threshold=0.5,
            keep_recent_turns=1,
        )
        script = [
            tool_turn("run_bash", {"command": f"python noisy.py {index}"})
            + [StreamFinished("tool_calls")]
            for index in range(8)
        ] + [text_turn("done")]

        agent, sink, provider = make_agent(config, script)
        agent.context.summarizer = None
        result = agent.run_turn("generate a lot of output")

        compactions = sink.of_type(CompactionDone)
        assert compactions
        assert result.completed
        assert provider.pairing_is_valid(), "compaction broke call/result pairing"

    def test_the_task_survives_compaction(self, project: Path) -> None:
        (project / "noisy.py").write_text(
            "for i in range(200): print('line', i, 'y' * 40)\n", encoding="utf-8"
        )
        config = auto(
            project,
            allow_outside_workspace=True,
            context_window=4000,
            compact_threshold=0.5,
            keep_recent_turns=1,
        )
        script = [
            tool_turn("run_bash", {"command": f"python noisy.py {index}"})
            + [StreamFinished("tool_calls")]
            for index in range(8)
        ] + [text_turn("done")]

        agent, _, provider = make_agent(config, script)
        agent.context.summarizer = None
        result = agent.run_turn("REMEMBER THIS TASK: tidy the noisy output")
        assert result.completed
        assert "REMEMBER THIS TASK" in agent.context.history[0].text

    def test_the_repo_map_is_rebuilt_after_the_agent_writes(self, project: Path) -> None:
        # A map that still describes the tree as it was is worse than none.
        agent, sink, provider = make_agent(
            auto(project),
            [
                tool_turn(
                    "write_file",
                    {"path": "brand_new_module.py", "content": "def freshly_added(): pass\n"},
                )
                + [StreamFinished("tool_calls")],
                text_turn("created it"),
            ],
        )
        agent.run_turn("create a module")
        assert "freshly_added" not in provider.systems[0]
        assert "freshly_added" in provider.systems[-1]
        assert [event.message for event in sink.of_type(Activity)] == [
            "Preparing task context",
            "Waiting for agent response",
            "Refreshing repo map",
            "Waiting for agent response",
        ]


class TestReporting:
    def test_the_closing_event_totals_the_run(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"})
                + [UsageReport(Usage(100, 20)), StreamFinished("tool_calls")],
                text_turn("done", usage=Usage(120, 25)),
            ],
        )
        agent.run_turn("read it")
        event = agent.finish("finished", trace_path="/tmp/t.jsonl")
        assert event.steps == 2
        assert event.usage.prompt_tokens == 220
        assert event.trace_path == "/tmp/t.jsonl"
        assert sink.of_type(RunFinished)

    def test_the_opening_event_describes_the_session(self, project: Path) -> None:
        agent, sink, _ = make_agent(auto(project), [text_turn("hi")])
        agent.announce("do the thing")
        (event,) = sink.of_type(RunStarted)
        assert event.task == "do the thing"
        assert "read_file" in event.tool_names
        assert event.system_tokens > 0

    def test_each_turn_reports_only_its_own_cost(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project),
            [text_turn("first", usage=Usage(100, 10)), text_turn("second", usage=Usage(200, 20))],
        )
        agent.run_turn("one")
        second = agent.run_turn("two")
        assert second.usage.prompt_tokens == 200
        assert [e.usage.prompt_tokens for e in sink.of_type(TurnFinished)] == [100, 200]

    def test_step_events_report_growing_prompt_size(self, project: Path) -> None:
        agent, sink, _ = make_agent(
            auto(project),
            [
                tool_turn("read_file", {"path": "calc.py"}) + [StreamFinished("tool_calls")],
                text_turn("done"),
            ],
        )
        agent.run_turn("read it")
        sizes = [event.prompt_tokens_estimate for event in sink.of_type(StepStarted)]
        assert sizes == sorted(sizes) and sizes[0] > 0

    def test_finished_steps_carry_the_assembled_message(self, project: Path) -> None:
        agent, sink, _ = make_agent(auto(project), [text_turn("the answer")])
        agent.run_turn("ask")
        (event,) = sink.of_type(StepFinished)
        assert event.message.text == "the answer"
        assert event.finish_reason == "stop"
        assert event.latency_s >= 0

    def test_the_user_turn_is_echoed_as_an_event(self, project: Path) -> None:
        agent, sink, _ = make_agent(auto(project), [text_turn("ok")])
        agent.run_turn("the request")
        assert [event.text for event in sink.of_type(UserMessage)] == ["the request"]
