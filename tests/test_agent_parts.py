"""Context management, loop guards, approval policy, repo map, and tracing.

The context tests carry the most weight. Compaction is the one place where a
plausible-looking implementation produces a request the provider rejects, so the
call/result pairing invariant is asserted after every operation rather than
assumed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from cagent.agent import repomap as repomap_module
from cagent.agent.approval import ApprovalPolicy, Decision
from cagent.agent.context import ContextManager
from cagent.agent.events import (
    ApprovalDecided,
    CollectingSink,
    FanOutSink,
    RunFinished,
    StepStarted,
    ToolFinished,
    ToolStarted,
    UserMessage,
    Warning,
)
from cagent.agent.guards import LoopGuard, call_signature
from cagent.agent.prompt import PromptBuilder
from cagent.agent.repomap import RepoMapIndex, build_repo_map
from cagent.agent.trace import TraceWriter, history_from_trace, read_trace
from cagent.config import AgentConfig
from cagent.errors import RepetitionDetected, TokenBudgetExceeded
from cagent.tools.base import ApprovalRequest, ToolOutcome
from cagent.types import (
    Message,
    RiskLevel,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)


def assert_pairing(manager: ContextManager) -> None:
    """Every tool call has exactly one result, in order.

    The invariant both wire formats enforce. Violating it does not degrade the
    run, it ends it with a provider 400.
    """
    calls = [part.id for m in manager.history for part in m.tool_calls]
    results = [part.call_id for m in manager.history for part in m.tool_results]
    assert calls == results, (calls, results)


def loaded(config: AgentConfig, steps: int, *, payload_lines: int = 60) -> ContextManager:
    """A transcript of ``steps`` tool-using steps, sized to force compaction."""
    manager = ContextManager(config)
    manager.set_overhead(system_tokens=400)
    manager.append(Message.user("do a lot of work"))
    for n in range(steps):
        manager.append(
            Message.assistant(
                ToolCallPart(id=f"c{n}", name="run_bash", arguments={"command": f"step {n}"})
            )
        )
        manager.append(
            Message.from_tool_results(
                [ToolResultPart(f"c{n}", "output line with content\n" * payload_lines)]
            )
        )
    return manager


@pytest.fixture
def tight(tmp_path: Path) -> AgentConfig:
    """A config whose window is small enough to exercise compaction quickly."""
    return AgentConfig(
        workspace=tmp_path,
        api_key="k",
        context_window=3000,
        compact_threshold=0.5,
        keep_recent_turns=2,
    )


class TestContextSegmentation:
    def test_a_step_is_one_block(self, tight: AgentConfig) -> None:
        # Grouping by user turn instead would make an agentic run a single
        # indivisible block, and compaction could never free anything.
        manager = loaded(tight, 4)
        assert len(manager.blocks()) == 5  # the task, plus one per step

    def test_an_assistant_turn_stays_with_its_results(self, tight: AgentConfig) -> None:
        manager = loaded(tight, 3)
        for block in manager.blocks()[1:]:
            roles = [m.role for m in block.messages]
            assert roles == ["assistant", "tool"], roles

    def test_parallel_results_share_their_assistant_turn(self, tight: AgentConfig) -> None:
        manager = ContextManager(tight)
        manager.append(Message.user("task"))
        manager.append(
            Message.assistant(
                ToolCallPart(id="a", name="read_file", arguments={}),
                ToolCallPart(id="b", name="read_file", arguments={}),
            )
        )
        manager.append(
            Message.from_tool_results([ToolResultPart("a", "one"), ToolResultPart("b", "two")])
        )
        assert len(manager.blocks()) == 2


class TestContextCompaction:
    def test_no_compaction_below_the_threshold(self, tight: AgentConfig) -> None:
        manager = loaded(tight, 1, payload_lines=1)
        assert not manager.needs_compaction()
        assert manager.compact().strategy == "none"

    def test_threshold_is_detected(self, tight: AgentConfig) -> None:
        assert loaded(tight, 8).needs_compaction()

    def test_eliding_tool_output_is_tried_first(self, tight: AgentConfig) -> None:
        # The cheapest stage, and usually sufficient, because tool output is the
        # bulk of a transcript.
        manager = loaded(tight, 3)
        report = manager.compact()
        assert report.strategy == "elide"
        assert report.tokens_after < report.tokens_before
        assert_pairing(manager)

    def test_elision_keeps_a_reminder_of_what_the_call_returned(
        self, tight: AgentConfig
    ) -> None:
        manager = loaded(tight, 3)
        manager.compact()
        contents = [p.content for m in manager.history for p in m.tool_results]
        assert any("removed to save context" in c for c in contents)
        assert any("output line" in c for c in contents)

    def test_heavy_pressure_escalates_and_reports_every_stage(
        self, tight: AgentConfig
    ) -> None:
        manager = loaded(replace(tight, context_window=2000), 10)
        report = manager.compact()
        assert "elide" in report.strategy
        assert len(report.stages) > 1, report.strategy
        # The protected task and recent step may keep us above the preferred
        # threshold; this is still usable because it fits the actual window.
        assert report.tokens_after <= manager.config.context_window
        assert_pairing(manager)

    def test_the_original_task_always_survives(self, tight: AgentConfig) -> None:
        # An agent that forgets the task finishes the wrong job confidently.
        manager = loaded(replace(tight, context_window=2000), 12)
        manager.compact()
        assert "do a lot of work" in manager.history[0].text

    def test_recent_steps_are_kept_verbatim(self, tight: AgentConfig) -> None:
        manager = loaded(replace(tight, context_window=2000), 10)
        manager.compact()
        recent = [p.content for m in manager.history[-2:] for p in m.tool_results]
        assert any("removed to save context" not in c for c in recent)

    def test_dropping_leaves_a_marker(self, tight: AgentConfig) -> None:
        # Without it the model sees the task jump straight to recent output and
        # redoes finished work.
        manager = loaded(replace(tight, context_window=2000), 12)
        manager.compact()
        joined = "\n".join(m.text for m in manager.history)
        assert "were dropped" in joined and "Re-read any file" in joined

    def test_a_summarizer_replaces_history_with_its_note(self, tight: AgentConfig) -> None:
        manager = loaded(replace(tight, context_window=2000), 10)
        manager.summarizer = lambda messages: "Ran 10 commands; app.py still needs the fix."
        report = manager.compact()
        joined = "\n".join(m.text for m in manager.history)
        assert "summarise" in report.strategy
        assert "Ran 10 commands" in joined and "Progress so far" in joined
        assert_pairing(manager)

    def test_the_summarizer_sees_only_the_history_being_replaced(
        self, tight: AgentConfig
    ) -> None:
        seen: list[int] = []

        def summarise(messages) -> str:
            seen.append(len(messages))
            return "note"

        manager = loaded(replace(tight, context_window=2000), 10)
        manager.summarizer = summarise
        manager.compact()
        assert seen and 0 < seen[0] < len(manager.history) + seen[0]

    def test_a_failing_summarizer_falls_back_to_dropping(self, tight: AgentConfig) -> None:
        def broken(messages) -> str:
            raise RuntimeError("summariser offline")

        manager = loaded(replace(tight, context_window=2000), 10)
        manager.summarizer = broken
        report = manager.compact()
        assert report.changed and "drop" in report.strategy
        assert_pairing(manager)

    def test_a_useless_summary_is_rejected(self, tight: AgentConfig) -> None:
        # Paying for a summarisation call that frees nothing is worse than
        # dropping the blocks outright.
        manager = loaded(tight, 10)
        manager.summarizer = lambda messages: "x" * 20_000
        report = manager.compact()
        assert "summarise" not in report.strategy

    def test_an_empty_summary_is_rejected(self, tight: AgentConfig) -> None:
        manager = loaded(tight, 10)
        manager.summarizer = lambda messages: "   "
        assert "summarise" not in manager.compact().strategy

    def test_repeated_compaction_converges_without_losing_validity(
        self, tight: AgentConfig
    ) -> None:
        manager = loaded(tight, 12)
        for _ in range(5):
            report = manager.compact()
            assert_pairing(manager)
            if not report.changed:
                break
        assert manager.history, "compaction emptied the transcript"
        assert "do a lot of work" in manager.history[0].text

    def test_pressure_is_reported_as_a_fraction(self, tight: AgentConfig) -> None:
        manager = loaded(tight, 4)
        assert manager.pressure > 0


class TestLoopGuard:
    def test_steps_are_counted(self, config: AgentConfig) -> None:
        guard = LoopGuard(config)
        guard.before_step()
        guard.before_step()
        assert guard.steps == 2

    def test_steps_are_unbounded_without_a_token_budget(self, tmp_path: Path) -> None:
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k"))
        for _ in range(45):
            guard.before_step()
        assert guard.steps == 45

    def test_the_token_budget_stops_the_run(self, tmp_path: Path) -> None:
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k", token_budget=100))
        guard.before_step()
        guard.add_tokens(150)
        with pytest.raises(TokenBudgetExceeded):
            guard.before_step()

    def test_no_budget_means_no_token_limit(self, config: AgentConfig) -> None:
        guard = LoopGuard(config)
        guard.add_tokens(10_000_000)
        guard.before_step()  # does not raise

    def test_a_repeat_earns_a_nudge_before_a_stop(self, tmp_path: Path) -> None:
        # The usual cause is a model that cannot see its own pattern, and one
        # sentence of feedback often fixes it.
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k", max_repeated_calls=3))
        call = ToolCallPart(id="c", name="read_file", arguments={"path": "a.py"})
        assert guard.check_call(call) is None
        nudge = guard.check_call(call)
        assert nudge is not None and "identical arguments" in nudge

    def test_the_nudge_is_not_repeated(self, tmp_path: Path) -> None:
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k", max_repeated_calls=5))
        call = ToolCallPart(id="c", name="read_file", arguments={"path": "a.py"})
        guard.check_call(call)
        assert guard.check_call(call) is not None
        assert guard.check_call(call) is None

    def test_an_ignored_nudge_ends_the_run(self, tmp_path: Path) -> None:
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k", max_repeated_calls=2))
        call = ToolCallPart(id="c", name="read_file", arguments={"path": "a.py"})
        with pytest.raises(RepetitionDetected):
            for _ in range(5):
                guard.check_call(call)

    def test_different_arguments_are_not_a_repeat(self, config: AgentConfig) -> None:
        guard = LoopGuard(config)
        for n in range(10):
            assert guard.check_call(
                ToolCallPart(id="c", name="read_file", arguments={"offset": n})
            ) is None

    def test_reordered_json_keys_still_count_as_the_same_call(self) -> None:
        first = ToolCallPart(id="a", name="edit", arguments={"path": "x", "old": "y"})
        second = ToolCallPart(id="b", name="edit", arguments={"old": "y", "path": "x"})
        assert call_signature(first) == call_signature(second)

    def test_progress_resets_repetition_tracking(self, tmp_path: Path) -> None:
        # A call repeated in a later turn at the user's request is not a live-lock.
        guard = LoopGuard(AgentConfig(workspace=tmp_path, api_key="k", max_repeated_calls=2))
        call = ToolCallPart(id="c", name="read_file", arguments={"path": "a.py"})
        guard.check_call(call)
        guard.check_call(call)
        guard.note_progress()
        assert guard.check_call(call) is None

def request(risk: RiskLevel, *, signature: str = "sig") -> ApprovalRequest:
    return ApprovalRequest(tool="t", risk=risk, summary="do a thing", signature=signature)


class TestApprovalPolicy:
    def test_no_request_means_no_permission_needed(self, config: AgentConfig) -> None:
        assert ApprovalPolicy(config).decide(None).approved

    def test_safe_calls_run_in_every_mode(self, tmp_path: Path) -> None:
        for mode in ("suggest", "auto-edit", "full-auto"):
            config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode=mode)  # type: ignore[arg-type]
            policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=False))
            assert policy.decide(request(RiskLevel.SAFE)).approved

    def test_suggest_mode_asks_about_a_mutation(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda r: (asked.append(r), Decision(approved=True))[1]
        )
        assert policy.decide(request(RiskLevel.MUTATING)).approved
        assert len(asked) == 1

    def test_full_auto_runs_mutations_unattended(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="full-auto")
        policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=False))
        assert policy.decide(request(RiskLevel.MUTATING)).approved

    def test_forced_confirmation_still_asks_in_full_auto(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="full-auto")
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda item: (asked.append(item), Decision(approved=False))[1]
        )
        sync = ApprovalRequest(
            tool="sandbox_sync",
            risk=RiskLevel.MUTATING,
            summary="copy changes back",
            always_prompt=True,
        )

        assert not policy.decide(sync).approved
        assert asked == [sync]

    @pytest.mark.parametrize(
        "tool_name", ["write_file", "edit_file", "multi_edit", "apply_patch"]
    )
    def test_auto_edit_runs_file_writes_unattended(
        self, tmp_path: Path, tool_name: str
    ) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="auto-edit")
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda r: (asked.append(r), Decision(approved=False))[1]
        )
        edit = ApprovalRequest(
            tool=tool_name,
            risk=RiskLevel.MUTATING,
            summary="edit a file",
        )

        assert policy.decide(edit).approved
        assert not asked

    def test_auto_edit_still_asks_for_mutating_shell_commands(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="auto-edit")
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda r: (asked.append(r), Decision(approved=False))[1]
        )
        shell = ApprovalRequest(
            tool="run_bash",
            risk=RiskLevel.MUTATING,
            summary="run tests",
        )

        assert not policy.decide(shell).approved
        assert asked == [shell]

    def test_no_mode_auto_approves_a_dangerous_call(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="full-auto")
        asked: list[ApprovalRequest] = []
        policy = ApprovalPolicy(
            config, prompter=lambda r: (asked.append(r), Decision(approved=True))[1]
        )
        policy.decide(request(RiskLevel.DANGEROUS))
        assert len(asked) == 1

    def test_a_remembered_answer_is_reused(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        calls = {"n": 0}

        def prompter(r: ApprovalRequest) -> Decision:
            calls["n"] += 1
            return Decision(approved=True, remember=True)

        policy = ApprovalPolicy(config, prompter=prompter)
        policy.decide(request(RiskLevel.MUTATING, signature="run_bash:pytest"))
        policy.decide(request(RiskLevel.MUTATING, signature="run_bash:pytest"))
        assert calls["n"] == 1

    def test_a_dangerous_answer_is_never_remembered(self, tmp_path: Path) -> None:
        # Consent must not transfer from one irreversible action to another.
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        calls = {"n": 0}

        def prompter(r: ApprovalRequest) -> Decision:
            calls["n"] += 1
            return Decision(approved=True, remember=True)

        policy = ApprovalPolicy(config, prompter=prompter)
        policy.decide(request(RiskLevel.DANGEROUS, signature="rm -rf a"))
        policy.decide(request(RiskLevel.DANGEROUS, signature="rm -rf a"))
        assert calls["n"] == 2
        assert not policy.remembered

    def test_a_remembered_refusal_is_also_reused(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=False, remember=True))
        policy.decide(request(RiskLevel.MUTATING))
        assert not policy.decide(request(RiskLevel.MUTATING)).approved
        assert policy.asked == 1

    def test_without_a_prompter_consent_is_refused(self, tmp_path: Path) -> None:
        # Non-interactive: refusing is the only safe answer, and the model is
        # told so as something it can work around.
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        policy = ApprovalPolicy(config, prompter=None)
        assert not policy.decide(request(RiskLevel.MUTATING)).approved

    def test_abort_is_recorded(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=False, abort=True))
        policy.decide(request(RiskLevel.MUTATING))
        assert policy.aborted

    def test_requires_prompt_matches_what_decide_does(self, tmp_path: Path) -> None:
        # The UI announces the question using this, so a disagreement would show
        # a prompt that never arrives.
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="auto-edit")
        policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=True))
        assert policy.requires_prompt(None) is False
        assert policy.requires_prompt(request(RiskLevel.SAFE)) is False
        assert policy.requires_prompt(request(RiskLevel.MUTATING)) is True
        edit = ApprovalRequest(
            tool="edit_file", risk=RiskLevel.MUTATING, summary="edit a file"
        )
        assert policy.requires_prompt(edit) is False
        policy.decide(request(RiskLevel.MUTATING, signature="sig"))
        policy.remembered["sig"] = True
        assert policy.requires_prompt(request(RiskLevel.MUTATING, signature="sig")) is False

    def test_describe_summarises_supervision(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", approval_mode="suggest")
        policy = ApprovalPolicy(config, prompter=lambda r: Decision(approved=True))
        policy.decide(request(RiskLevel.MUTATING))
        assert "suggest" in policy.describe()


class TestRepoMap:
    def test_python_signatures_are_exact(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text(
            "import os\n\n\n"
            "def compute(a: int, b: str = 'x', *rest, key: float | None = None) -> bool:\n"
            "    return True\n\n\n"
            "class Widget(Base, Mixin):\n"
            "    def render(self, deep: bool = False) -> str:\n"
            "        return ''\n\n"
            "    def _private(self):\n"
            "        pass\n",
            encoding="utf-8",
        )
        result = build_repo_map(tmp_path, token_budget=2000)
        assert "def compute(a: int, b: str, *rest, key: float | None) -> bool" in result.text
        assert "class Widget(Base, Mixin)" in result.text
        assert "def render(self, deep: bool) -> str" in result.text
        assert "_private" not in result.text  # private members are noise here

    def test_module_constants_are_included(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text("TIMEOUT = 30\nlowercase = 1\n", encoding="utf-8")
        result = build_repo_map(tmp_path, token_budget=2000)
        assert "TIMEOUT" in result.text and "lowercase" not in result.text

    def test_a_syntax_error_still_yields_something(self, tmp_path: Path) -> None:
        # Normal state for a file the agent is midway through editing.
        (tmp_path / "broken.py").write_text("def half_written(\nclass Thing:\n", encoding="utf-8")
        result = build_repo_map(tmp_path, token_budget=2000)
        assert "Thing" in result.text

    def test_other_languages_are_recognised(self, tmp_path: Path) -> None:
        (tmp_path / "a.ts").write_text(
            "export function go(x: number) {}\nexport interface Shape {}\n", encoding="utf-8"
        )
        (tmp_path / "b.go").write_text("func Handle(w int) {}\ntype Server struct {}\n", "utf-8")
        result = build_repo_map(tmp_path, token_budget=2000)
        assert "go" in result.text and "Shape" in result.text
        assert "Handle" in result.text and "Server" in result.text

    @pytest.mark.parametrize(
        ("filename", "source", "declaration"),
        [
            ("service.cs", "public class Worker {}\n", "Worker"),
            ("worker.swift", "struct Worker {}\n", "Worker"),
            ("worker.ex", "defmodule Demo.Worker do\nend\n", "Demo.Worker"),
            ("worker.lua", "function worker.run()\nend\n", "worker.run"),
            ("schema.sql", "CREATE TABLE widgets (id INT);\n", "CREATE TABLE"),
            ("panel.vue", "<script>\nfunction openPanel() {}\n</script>\n", "openPanel"),
        ],
    )
    def test_fallback_parser_covers_multiple_language_families(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        filename: str,
        source: str,
        declaration: str,
    ) -> None:
        monkeypatch.setattr(repomap_module, "_tree_sitter_parser", lambda language: None)
        (tmp_path / filename).write_text(source, encoding="utf-8")

        result = build_repo_map(tmp_path, token_budget=1000)

        assert filename in result.text
        assert declaration in result.text
        assert result.outlines[0].parser == "fallback"

    def test_a_file_without_declarations_still_contributes_its_path(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "settings.ts").write_text("const timeout = 30;\n", encoding="utf-8")

        result = build_repo_map(tmp_path, token_budget=1000)

        assert "settings.ts" in result.text
        assert result.files_included == 1

    def test_a_large_file_keeps_metadata_without_loading_its_body(
        self, tmp_path: Path
    ) -> None:
        large = tmp_path / "bundle.js"
        large.write_bytes((b"x = 1;\n" * 60_000) + b"tail")

        result = build_repo_map(tmp_path, token_budget=1000)

        assert "bundle.js (60001 lines, javascript)" in result.text
        assert result.outlines[0].parser == "metadata"

    def test_the_current_task_ranks_matching_paths_and_symbols_first(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "main.py").write_text(
            "".join(f"def generic_{index}(): pass\n" for index in range(20)),
            encoding="utf-8",
        )
        target = tmp_path / "deep" / "transport"
        target.mkdir(parents=True)
        (target / "stream_controller.ts").write_text(
            "export function cancelProviderStream() {}\n",
            encoding="utf-8",
        )

        result = build_repo_map(
            tmp_path,
            token_budget=1000,
            query="repair provider stream cancellation",
        )

        assert result.outlines[0].path == "deep/transport/stream_controller.ts"

    def test_a_small_budget_preserves_paths_then_adds_relevant_detail(
        self, tmp_path: Path
    ) -> None:
        for index in range(12):
            (tmp_path / f"module_{index}.py").write_text(
                "".join(f"def helper_{index}_{item}(): pass\n" for item in range(10)),
                encoding="utf-8",
            )
        (tmp_path / "interrupt_handler.py").write_text(
            "def cancel_stream(): pass\n", encoding="utf-8"
        )

        result = build_repo_map(
            tmp_path,
            token_budget=240,
            query="cancel stream interrupt",
        )

        assert result.tokens <= 240
        assert len(result.outlines) > 1
        assert "def cancel_stream" in result.text
        assert any(
            block.count("\n") == 0
            for block in result.text.split("\nmodule_")
            if ".py (" in block
        )

    def test_incremental_refresh_only_reparses_changed_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = tmp_path / "first.py"
        second = tmp_path / "second.py"
        first.write_text("def first(): pass\n", encoding="utf-8")
        second.write_text("def second(): pass\n", encoding="utf-8")
        original = repomap_module._outline_file
        parsed: list[str] = []

        def tracking_outline(path: Path, workspace: Path, language: str):
            parsed.append(path.name)
            return original(path, workspace, language)

        monkeypatch.setattr(repomap_module, "_outline_file", tracking_outline)
        index = RepoMapIndex(tmp_path)
        index.refresh()
        assert sorted(parsed) == ["first.py", "second.py"]

        parsed.clear()
        index.refresh()
        assert parsed == []

        first.write_text("def first_changed(value): pass\n", encoding="utf-8")
        index.refresh()
        assert parsed == ["first.py"]
        assert "first_changed" in index.render(token_budget=1000).text

    def test_missing_tree_sitter_grammar_never_calls_get_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pack = ModuleType("tree_sitter_language_pack")
        fake_pack.downloaded_languages = lambda: []  # type: ignore[attr-defined]
        calls: list[str] = []

        def get_parser(language: str) -> object:
            calls.append(language)
            raise AssertionError("get_parser may download a grammar")

        fake_pack.get_parser = get_parser  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake_pack)
        repomap_module._tree_sitter_parser.cache_clear()

        assert repomap_module._tree_sitter_parser("typescript") is None
        assert calls == []
        repomap_module._tree_sitter_parser.cache_clear()

    def test_generated_directories_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("def kept(): pass\n", encoding="utf-8")
        vendored = tmp_path / "node_modules"
        vendored.mkdir()
        (vendored / "dep.py").write_text("def vendored(): pass\n", encoding="utf-8")
        result = build_repo_map(tmp_path, token_budget=2000)
        assert "kept" in result.text and "vendored" not in result.text

    def test_the_budget_is_respected_and_the_omission_reported(self, tmp_path: Path) -> None:
        for n in range(60):
            (tmp_path / f"mod{n}.py").write_text(
                "".join(f"def function_{n}_{k}(argument_name): pass\n" for k in range(12)),
                encoding="utf-8",
            )
        result = build_repo_map(tmp_path, token_budget=300)
        assert result.tokens <= 300
        assert result.truncated
        assert any("omitted" in note for note in result.notes)

    def test_shallow_files_outrank_deep_ones(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def entry(): pass\n", encoding="utf-8")
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "buried.py").write_text("def buried(): pass\n", encoding="utf-8")
        result = build_repo_map(tmp_path, token_budget=2000)
        assert result.text.index("main.py") < result.text.index("buried.py")

    def test_an_empty_project_yields_an_empty_map(self, tmp_path: Path) -> None:
        assert build_repo_map(tmp_path, token_budget=1000).is_empty()

    def test_a_zero_budget_yields_an_empty_map(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text("def f(): pass\n", encoding="utf-8")
        assert build_repo_map(tmp_path, token_budget=0).is_empty()


class TestPromptBuilder:
    def test_the_prompt_names_the_environment_and_tools(self, config: AgentConfig) -> None:
        from cagent.tools.registry import default_registry

        specs = tuple(default_registry().specs())
        prompt = PromptBuilder(config).build(tools=specs)
        assert str(config.workspace) in prompt.text
        assert "edit_file" in prompt.text and "run_bash" in prompt.text
        assert prompt.tokens > 0

    def test_the_repo_map_is_included_when_there_is_one(self, config: AgentConfig) -> None:
        (config.workspace / "mod.py").write_text("def discoverable(): pass\n", encoding="utf-8")
        prompt = PromptBuilder(config).build()
        assert "discoverable" in prompt.text
        assert prompt.repo_map is not None

    def test_the_map_can_be_switched_off(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, api_key="k", repo_map_enabled=False)
        (tmp_path / "mod.py").write_text("def hidden(): pass\n", encoding="utf-8")
        prompt = PromptBuilder(config).build()
        assert "hidden" not in prompt.text and prompt.repo_map is None

    def test_the_map_is_cached_then_rebuilt_on_request(self, config: AgentConfig) -> None:
        # A map that still claims a file has no main() after one was added is
        # worse than no map.
        builder = PromptBuilder(config)
        (config.workspace / "mod.py").write_text("def first(): pass\n", encoding="utf-8")
        assert "first" in builder.build().text

        (config.workspace / "mod.py").write_text("def second(): pass\n", encoding="utf-8")
        assert "second" not in builder.build().text  # served from cache

        builder.invalidate_map()
        assert "second" in builder.build().text

    def test_the_map_follows_a_workspace_switch(self, config: AgentConfig, tmp_path: Path) -> None:
        host_file = config.workspace / "host.py"
        host_file.write_text("def host_symbol(): pass\n", encoding="utf-8")
        builder = PromptBuilder(config)
        assert "host_symbol" in builder.build().text

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "snapshot.py").write_text(
            "def snapshot_symbol(): pass\n", encoding="utf-8"
        )
        builder.workspace = snapshot
        builder.invalidate_map()

        prompt = builder.build()
        assert "snapshot_symbol" in prompt.text
        assert "host_symbol" not in prompt.text

    def test_project_markers_are_advertised(self, config: AgentConfig) -> None:
        (config.workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        assert "pyproject.toml" in PromptBuilder(config).build().text

    def test_sandbox_prompt_names_container_runtime(self, config: AgentConfig) -> None:
        snapshot = config.workspace / "snapshot"
        snapshot.mkdir()
        prompt = PromptBuilder(config, workspace=snapshot).build()
        assert "Linux Docker container" in prompt.text
        assert "python3/python" in prompt.text
        assert "never Windows" in prompt.text
        assert "Host platform" in prompt.text and "Host Python" in prompt.text

    def test_extra_context_is_appended_verbatim(self, config: AgentConfig) -> None:
        prompt = PromptBuilder(config).build(extra_context="Always use tabs.")
        assert "Always use tabs." in prompt.text


class TestTrace:
    def test_history_from_trace_preserves_tool_call_order(self) -> None:
        records = [
            {"type": "user", "text": "inspect both files"},
            {
                "type": "step_finished",
                "message": {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "a",
                            "name": "read_file",
                            "arguments": {"path": "a.py"},
                        },
                        {
                            "type": "tool_call",
                            "id": "b",
                            "name": "read_file",
                            "arguments": {"path": "b.py"},
                        },
                    ],
                },
            },
            {"type": "tool_finished", "id": "b", "content": "B"},
            {"type": "tool_finished", "id": "a", "content": "A"},
            {
                "type": "step_finished",
                "message": {"role": "assistant", "parts": [{"type": "text", "text": "done"}]},
            },
        ]

        history = history_from_trace(records)

        assert [message.role for message in history] == ["user", "assistant", "tool", "assistant"]
        assert [result.call_id for result in history[2].tool_results] == ["a", "b"]
        assert history[-1].text == "done"

    def test_history_from_trace_drops_an_incomplete_final_tool_call(self) -> None:
        records = [
            {"type": "user", "text": "run the check"},
            {
                "type": "step_finished",
                "message": {
                    "role": "assistant",
                    "parts": [
                        {"type": "thinking", "text": "plan", "signature": "sig"},
                        {
                            "type": "tool_call",
                            "id": "a",
                            "name": "run_bash",
                            "arguments": {"command": "pytest"},
                        },
                    ],
                },
            },
        ]

        history = history_from_trace(records)

        assert [message.role for message in history] == ["user"]

    def test_history_from_trace_drops_incomplete_call_before_a_new_user_turn(self) -> None:
        records = [
            {"type": "user", "text": "first"},
            {
                "type": "step_finished",
                "message": {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "a",
                            "name": "run_bash",
                            "arguments": {"command": "pytest"},
                        }
                    ],
                },
            },
            {"type": "user", "text": "second"},
        ]

        history = history_from_trace(records)

        assert [message.role for message in history] == ["user", "user"]
        assert [message.text for message in history] == ["first", "second"]

    def test_events_are_recorded_as_jsonl(self, config: AgentConfig) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="abc")
        assert writer is not None
        writer.handle(UserMessage("a real task"))
        call = ToolCallPart(id="c", name="read_file", arguments={"path": "a.py"})
        writer.handle(StepStarted(step=1, prompt_tokens_estimate=500))
        writer.handle(ToolStarted(call=call, risk=RiskLevel.SAFE))
        writer.handle(
            ToolFinished(
                call=call,
                outcome=ToolOutcome.ok("1\tcontents", metadata={"path": "a.py"}),
                duration_s=0.01,
            )
        )
        writer.handle(RunFinished(reason="finished", steps=1, usage=Usage(10, 5), elapsed_s=1.0))
        writer.close()

        records = read_trace(writer.path)
        by_kind = {record["type"]: record for record in records}
        assert list(by_kind) == [
            "session",
            "user",
            "step_started",
            "tool_started",
            "tool_finished",
            "run_finished",
        ]
        # Arguments are recorded once, at the start, rather than on both events.
        assert by_kind["tool_started"]["arguments"] == {"path": "a.py"}
        assert by_kind["tool_finished"]["metadata"] == {"path": "a.py"}
        assert by_kind["tool_finished"]["is_error"] is False
        assert by_kind["run_finished"]["usage"]["prompt"] == 10

    def test_activity_events_are_transient_and_not_recorded(self, config: AgentConfig) -> None:
        from cagent.agent.events import Activity

        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="activity")
        assert writer is not None
        writer.handle(Activity("Building repo map"))
        writer.handle(UserMessage("a real task"))
        writer.close()

        records = read_trace(writer.path)
        assert all(record.get("type") != "activity" for record in records)

    def test_the_api_key_never_reaches_the_trace(self, config: AgentConfig) -> None:
        config.trace_dir = config.workspace / "traces"
        config.api_key = "sk-super-secret"
        writer = TraceWriter.create(config, session_id="abc")
        assert writer is not None
        writer.handle(UserMessage("a real task"))
        writer.handle(RunFinished(reason="x", steps=0, usage=Usage(), elapsed_s=0.0))
        writer.close()
        assert "sk-super-secret" not in writer.path.read_text(encoding="utf-8")

    def test_huge_values_are_clipped_with_a_note(self, config: AgentConfig) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="abc")
        assert writer is not None
        writer.handle(UserMessage("a real task"))
        writer.handle(
            ToolFinished(
                call=ToolCallPart(id="c", name="run_bash", arguments={}),
                outcome=ToolOutcome.ok("x" * 100_000),
                duration_s=0.1,
            )
        )
        writer.close()
        record = next(r for r in read_trace(writer.path) if r["type"] == "tool_finished")
        assert "omitted from trace" in record["content"]

    def test_deltas_are_not_recorded(self, config: AgentConfig) -> None:
        # They are re-derivable from the assembled message, and thousands of
        # fragments would bury the events that carry information.
        from cagent.agent.events import TextDelta

        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="abc")
        assert writer is not None
        before = writer.events_written
        writer.handle(TextDelta("a"))
        assert writer.events_written == before
        writer.close()

    def test_tracing_off_produces_no_writer(self, config: AgentConfig) -> None:
        config.trace_dir = None
        assert TraceWriter.create(config, session_id="abc") is None

    def test_empty_trace_can_be_discarded(self, config: AgentConfig) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="empty")
        assert writer is not None and not writer.path.exists()
        writer.discard_if_empty()
        assert not writer.path.exists()

    def test_trace_with_a_user_turn_is_kept_when_discarding_empty_sessions(
        self, config: AgentConfig
    ) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="used")
        assert writer is not None
        writer.handle(UserMessage("a real task"))
        writer.discard_if_empty()
        assert writer.path.exists()

    def test_an_unwritable_location_degrades_instead_of_raising(
        self, config: AgentConfig
    ) -> None:
        # Losing observability must not lose the user's task.
        blocker = config.workspace / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        config.trace_dir = blocker / "traces"
        writer = TraceWriter.create(config, session_id="abc")
        assert writer is not None
        writer.handle(UserMessage("a real task"))
        assert writer.error is not None
        writer.handle(RunFinished(reason="x", steps=0, usage=Usage(), elapsed_s=0.0))

    def test_a_truncated_final_line_is_tolerated(self, config: AgentConfig) -> None:
        path = config.workspace / "partial.jsonl"
        path.write_text(
            json.dumps({"type": "session"}) + "\n" + '{"type": "step_start',
            encoding="utf-8",
        )
        assert [r["type"] for r in read_trace(path)] == ["session"]

    def test_a_utf8_bom_is_tolerated(self, config: AgentConfig) -> None:
        path = config.workspace / "bom.jsonl"
        path.write_bytes(b'\xef\xbb\xbf{"type":"session"}\n')
        assert [r["type"] for r in read_trace(path)] == ["session"]

    def test_history_checkpoint_can_be_restored(self, config: AgentConfig) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="checkpoint")
        assert writer is not None
        writer.record_history([Message.user("old"), Message.assistant(TextPart("answer"))])
        assert not writer.path.exists()
        writer.handle(UserMessage("continue"))
        writer.close()

        history = history_from_trace(read_trace(writer.path))

        assert [message.text for message in history] == ["old", "answer", "continue"]

    def test_empty_history_checkpoint_keeps_an_undone_turn_removed(
        self, config: AgentConfig
    ) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="undo")
        assert writer is not None
        writer.handle(UserMessage("remove me"))
        writer.record_history([])
        writer.close()

        records = read_trace(writer.path)

        assert records[-1]["type"] == "history_checkpoint"
        assert records[-1]["messages"] == []
        assert history_from_trace(records) == []

    def test_restored_history_alone_is_discarded_as_an_empty_new_session(
        self, config: AgentConfig
    ) -> None:
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="resumed-only")
        assert writer is not None
        writer.record_history([Message.user("old")])
        assert not writer.has_user_message
        writer.discard_if_empty()
        assert not writer.path.exists()


class TestEventSinks:
    def test_collecting_sink_filters_by_type(self) -> None:
        sink = CollectingSink()
        sink.handle(Warning("careful"))
        sink.handle(StepStarted(step=1, prompt_tokens_estimate=10))
        assert len(sink.of_type(Warning)) == 1
        assert sink.of_type(StepStarted)[0].step == 1

    def test_fan_out_reaches_every_sink(self) -> None:
        one, two = CollectingSink(), CollectingSink()
        FanOutSink([one, two]).handle(Warning("hello"))
        assert one.events and two.events

    def test_a_broken_sink_is_dropped_not_propagated(self) -> None:
        # A failing renderer or an unwritable trace must not end the run.
        class Broken:
            def handle(self, event: object) -> None:
                raise RuntimeError("renderer died")

        good = CollectingSink()
        fan = FanOutSink([Broken(), good])
        fan.handle(Warning("first"))
        fan.handle(Warning("second"))
        assert len(good.events) == 2
        assert fan.failures and "renderer died" in fan.failures[0]


class TestApprovalEventShape:
    def test_automatic_decisions_are_flagged(self) -> None:
        event = ApprovalDecided(
            request=request(RiskLevel.SAFE), approved=True, automatic=True
        )
        assert event.automatic and event.approved
