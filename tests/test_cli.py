"""The command-line surface: parsing, rendering, pricing, replay.

Rendering is tested by writing to a captured console rather than a terminal, so
the assertions are about what information reaches the user — the diff, the exit
code, the cost — and not about escape codes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from cagent.agent.approval import Decision
from cagent.agent.events import (
    ApprovalDecided,
    CompactionDone,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    Warning,
)
from cagent.cli.app import _overrides, build_parser, main
from cagent.cli.pricing import Price, estimate_cost, parse_prices, price_for
from cagent.cli.render import ConsoleRenderer, prompt_for_approval
from cagent.tools.base import ApprovalRequest, ToolOutcome
from cagent.types import Message, RiskLevel, TextPart, ToolCallPart, Usage


@pytest.fixture
def captured() -> tuple[Console, io.StringIO]:
    """A console writing to a buffer, wide enough not to wrap assertions apart."""
    buffer = io.StringIO()
    return Console(file=buffer, width=200, highlight=False, soft_wrap=False), buffer


class TestArgumentParsing:
    def test_a_task_can_be_given_as_bare_words(self) -> None:
        args = build_parser().parse_args(["fix", "the", "bug"])
        assert args.task == ["fix", "the", "bug"]

    def test_no_task_means_interactive(self) -> None:
        assert build_parser().parse_args([]).task == []

    def test_flags_map_onto_config_fields(self) -> None:
        args = build_parser().parse_args(
            [
                "--base-url", "https://api.example.com/v1",
                "--model", "m",
                "--max-steps", "7",
                "task",
            ]
        )
        overrides = _overrides(args)
        assert overrides["base_url"] == "https://api.example.com/v1"
        assert overrides["model"] == "m"
        assert overrides["max_steps"] == 7

    def test_unset_flags_stay_none_so_lower_layers_decide(self) -> None:
        # The loader drops None, which is what lets a config file or environment
        # variable survive an unrelated flag being passed.
        overrides = _overrides(build_parser().parse_args(["task"]))
        assert overrides["base_url"] is None and overrides["model"] is None

    def test_no_key_marks_the_endpoint_as_keyless(self) -> None:
        overrides = _overrides(build_parser().parse_args(["--no-key", "task"]))
        assert overrides["requires_key"] is False

    def test_the_wire_can_be_selected(self) -> None:
        args = build_parser().parse_args(["--wire", "anthropic", "task"])
        assert _overrides(args)["wire"] == "anthropic"

    def test_there_is_no_provider_flag(self) -> None:
        # A named provider would mean a built-in table of vendor model names,
        # and those go stale; the endpoint is described directly instead.
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--provider", "openai", "task"])

    def test_yes_is_shorthand_for_full_auto(self) -> None:
        assert _overrides(build_parser().parse_args(["-y", "task"]))["approval_mode"] == "full-auto"

    def test_an_explicit_approval_mode_is_honoured(self) -> None:
        args = build_parser().parse_args(["--approval", "suggest", "task"])
        assert _overrides(args)["approval_mode"] == "suggest"

    def test_repo_map_can_be_switched_off(self) -> None:
        args = build_parser().parse_args(["--no-repo-map", "task"])
        assert _overrides(args)["repo_map_enabled"] is False

    def test_there_is_no_api_key_flag(self) -> None:
        # A key on the command line lands in shell history and the process list.
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--api-key", "sk-oops", "task"])

    def test_an_invalid_choice_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--approval", "yolo", "task"])


@pytest.fixture
def clean_env(monkeypatch) -> None:
    """No inherited endpoint settings.

    Without this, a real CAGENT_* or vendor key in the developer's shell would
    silently satisfy the very thing a test asserts is missing.
    """
    from cagent.config import VENDOR_KEY_VARIABLES

    for name in ("CAGENT_API_KEY", "CAGENT_BASE_URL", "CAGENT_MODEL", "CAGENT_WIRE"):
        monkeypatch.delenv(name, raising=False)
    for name in VENDOR_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


class TestInformationalCommands:
    def test_list_tools_prints_every_tool(self, capsys) -> None:
        assert main(["--list-tools"]) == 0
        out = capsys.readouterr().out
        for name in ("read_file", "edit_file", "run_bash", "grep_search"):
            assert name in out

    def test_list_tools_marks_optional_arguments(self, capsys) -> None:
        main(["--list-tools"])
        out = capsys.readouterr().out
        assert "offset?" in out  # optional
        assert "path," in out or "path " in out  # required, unmarked

    def test_show_config_reports_the_resolved_endpoint(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        monkeypatch.setenv("CAGENT_API_KEY", "sk-secret")
        monkeypatch.setenv("CAGENT_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("CAGENT_MODEL", "env-model")
        assert main(["--show-config", "--workspace", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "https://api.example.com/v1" in out
        assert "env-model" in out
        assert "set" in out

    def test_show_config_never_prints_the_key(self, capsys, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("CAGENT_API_KEY", "sk-do-not-print-me")
        main(["--show-config", "--workspace", str(tmp_path)])
        assert "sk-do-not-print-me" not in capsys.readouterr().out

    def test_show_config_works_when_nothing_is_configured(
        self, capsys, tmp_path, clean_env
    ) -> None:
        # This command is most useful precisely when the configuration is
        # incomplete, so it reports gaps rather than raising.
        assert main(["--show-config", "--workspace", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "not set" in out
        assert "CAGENT_BASE_URL" in out and "CAGENT_MODEL" in out

    def test_show_config_names_the_full_request_url(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        monkeypatch.setenv("CAGENT_API_KEY", "sk-x")
        main(
            [
                "--show-config",
                "--base-url", "https://api.example.com/v1/",
                "--model", "m",
                "--workspace", str(tmp_path),
            ]
        )
        assert "https://api.example.com/v1/chat/completions" in capsys.readouterr().out

    def test_show_config_names_the_anthropic_path(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        monkeypatch.setenv("CAGENT_API_KEY", "sk-x")
        main(
            [
                "--show-config",
                "--base-url", "https://api.example.com/v1",
                "--model", "m",
                "--wire", "anthropic",
                "--workspace", str(tmp_path),
            ]
        )
        assert "/v1/messages" in capsys.readouterr().out


class TestEndpointConfiguration:
    """An endpoint is four settings: base_url, model, api_key, wire."""

    def test_base_url_model_and_key_are_sufficient(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        monkeypatch.setenv("CAGENT_API_KEY", "sk-gateway")
        code = main(
            [
                "--show-config",
                "--base-url", "https://gw.example.com/v1",
                "--model", "some-gateway-model",
                "--workspace", str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "https://gw.example.com/v1" in out
        assert "some-gateway-model" in out

    def test_a_missing_endpoint_is_reported_before_any_request(
        self, capsys, tmp_path, clean_env
    ) -> None:
        code = main(["--model", "m", "--workspace", str(tmp_path), "do a thing"])
        out = capsys.readouterr().out
        assert code == 2
        assert "base_url" in out.lower() or "CAGENT_BASE_URL" in out

    def test_a_missing_model_is_reported_rather_than_guessed(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        # There is deliberately no default model: a name frozen at release time
        # fails later as an unhelpful remote 404.
        monkeypatch.setenv("CAGENT_API_KEY", "sk-x")
        code = main(
            [
                "--base-url", "https://gw.example.com/v1",
                "--workspace", str(tmp_path),
                "do a thing",
            ]
        )
        out = capsys.readouterr().out
        assert code == 2
        assert "model" in out.lower()

    def test_a_missing_key_names_the_endpoint(self, capsys, tmp_path, clean_env) -> None:
        code = main(
            [
                "--base-url", "https://gw.example.com/v1",
                "--model", "m",
                "--workspace", str(tmp_path),
                "do a thing",
            ]
        )
        out = capsys.readouterr().out
        assert code == 2
        assert "gw.example.com" in out

    def test_no_vendor_variable_is_suggested_over_the_endpoint(
        self, capsys, tmp_path, clean_env
    ) -> None:
        # Telling someone to set DEEPSEEK_API_KEY for their self-hosted gateway
        # sends them looking in the wrong place.
        main(
            [
                "--base-url", "https://gw.example.com/v1",
                "--model", "m",
                "--workspace", str(tmp_path),
                "do a thing",
            ]
        )
        out = capsys.readouterr().out
        assert "gw.example.com" in out
        assert "DEEPSEEK_API_KEY" not in out

    def test_the_hint_shows_a_complete_configuration(
        self, capsys, tmp_path, clean_env
    ) -> None:
        main(["--workspace", str(tmp_path), "do a thing"])
        out = capsys.readouterr().out
        for expected in ("CAGENT_BASE_URL", "CAGENT_MODEL", "CAGENT_API_KEY"):
            assert expected in out

    def test_no_key_permits_a_local_server(self, capsys, tmp_path, clean_env) -> None:
        code = main(
            [
                "--show-config",
                "--base-url", "http://localhost:11434/v1",
                "--model", "some-local-model",
                "--no-key",
                "--workspace", str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "not needed" in out

    def test_a_vendor_variable_is_still_picked_up(
        self, capsys, monkeypatch, tmp_path, clean_env
    ) -> None:
        # A convenience: an already-exported key need not be duplicated into
        # CAGENT_API_KEY. It says nothing about which endpoint is called.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-vendor-var")
        main(
            [
                "--show-config",
                "--base-url", "https://gw.example.com/v1",
                "--model", "m",
                "--workspace", str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert "set" in out
        assert "sk-from-vendor-var" not in out


class TestRendering:
    def test_the_header_names_the_model_endpoint_and_tools(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            RunStarted(
                task="fix it",
                model="some-model",
                endpoint="https://api.example.com/v1",
                system_tokens=1200,
                tool_names=("read_file", "edit_file"),
            )
        )
        out = buffer.getvalue()
        # The endpoint matters as much as the model: the same name can be
        # served by several of them.
        assert "some-model" in out
        assert "api.example.com" in out
        assert "read_file" in out

    def test_a_tool_call_is_shown_with_its_arguments(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ToolStarted(
                call=ToolCallPart(id="c", name="edit_file", arguments={"path": "app.py"}),
                risk=RiskLevel.MUTATING,
            )
        )
        out = buffer.getvalue()
        assert "edit_file" in out and "app.py" in out

    def test_a_diff_is_shown_when_a_tool_produces_one(self, config, captured) -> None:
        # The diff is the receipt: it is why the model need not narrate the change.
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ToolFinished(
                call=ToolCallPart(id="c", name="edit_file", arguments={"path": "app.py"}),
                outcome=ToolOutcome.ok(
                    "Edited app.py",
                    display="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old line\n+new line\n",
                    metadata={"added": 1, "removed": 1, "path": "app.py"},
                ),
                duration_s=0.02,
            )
        )
        out = buffer.getvalue()
        assert "-old line" in out and "+new line" in out

    def test_an_exit_code_is_summarised(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ToolFinished(
                call=ToolCallPart(id="c", name="run_bash", arguments={"command": "pytest"}),
                outcome=ToolOutcome.error(
                    "exit code: 1\n--- stderr ---\nassert 1 == 2",
                    metadata={"exit_code": 1},
                ),
                duration_s=1.5,
            )
        )
        out = buffer.getvalue()
        assert "exit 1" in out and "assert 1 == 2" in out

    def test_long_tool_output_is_clipped_in_the_display(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ToolFinished(
                call=ToolCallPart(id="c", name="run_bash", arguments={}),
                outcome=ToolOutcome.ok("\n".join(f"line {n}" for n in range(200))),
                duration_s=0.1,
            )
        )
        out = buffer.getvalue()
        assert "more lines" in out
        assert "line 199" not in out

    def test_a_step_reports_its_share_of_the_window(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(StepStarted(step=3, max_steps=40, prompt_tokens_estimate=64_000))
        out = buffer.getvalue()
        assert "step 3/40" in out and "64,000" in out and "50%" in out

    def test_compaction_is_reported_to_the_user(self, config, captured) -> None:
        # Silently shrinking history looks to the user like the model getting confused.
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            CompactionDone(
                strategy="elide+drop",
                tokens_before=90_000,
                tokens_after=40_000,
                messages_before=48,
                messages_after=12,
            )
        )
        out = buffer.getvalue()
        assert "elide+drop" in out and "90,000" in out and "40,000" in out

    def test_warnings_are_shown_even_when_quiet(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console, quiet=True)
        renderer.handle(Warning("the context window is nearly full"))
        assert "nearly full" in buffer.getvalue()

    def test_quiet_suppresses_routine_narration(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console, quiet=True)
        renderer.handle(StepStarted(step=1, max_steps=10, prompt_tokens_estimate=100))
        renderer.handle(
            ToolStarted(
                call=ToolCallPart(id="c", name="read_file", arguments={}), risk=RiskLevel.SAFE
            )
        )
        assert buffer.getvalue() == ""

    def test_an_automatic_approval_is_not_announced(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ApprovalDecided(
                request=ApprovalRequest(tool="t", risk=RiskLevel.SAFE, summary="s"),
                approved=True,
                automatic=True,
            )
        )
        assert buffer.getvalue() == ""

    def test_the_summary_totals_tokens_and_cost(self, config, captured) -> None:
        console, buffer = captured
        config.model = "some-model"
        config.prices = {"some-model": {"input_per_m": 1.0, "output_per_m": 4.0}}
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            RunFinished(
                reason="finished",
                steps=6,
                usage=Usage(prompt_tokens=50_000, completion_tokens=2_000, cached_tokens=10_000),
                elapsed_s=42.5,
                trace_path="/tmp/trace.jsonl",
            )
        )
        out = buffer.getvalue()
        assert "50,000" in out and "2,000" in out
        assert "42.5s" in out
        assert "$" in out
        assert "trace.jsonl" in out

    def test_no_configured_rate_says_so_instead_of_guessing(self, config, captured) -> None:
        console, buffer = captured
        config.model = "some-local-model"
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            RunFinished(reason="finished", steps=1, usage=Usage(10, 5), elapsed_s=1.0)
        )
        out = buffer.getvalue()
        assert "no rate set" in out
        assert "$" not in out

    def test_the_summary_lists_files_that_changed(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(
            ToolFinished(
                call=ToolCallPart(id="c", name="edit_file", arguments={}),
                outcome=ToolOutcome.ok("done", metadata={"path": "src/app.py"}),
                duration_s=0.1,
            )
        )
        renderer.handle(
            RunFinished(reason="finished", steps=1, usage=Usage(1, 1), elapsed_s=1.0)
        )
        assert "src/app.py" in buffer.getvalue()

    def test_a_rendering_fault_does_not_propagate(self, config, captured) -> None:
        # A display bug must not cost the user their task.
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)

        class Hostile:
            """An event whose fields raise when read."""

            @property
            def text(self) -> str:
                raise RuntimeError("bad event")

        renderer.handle(TextDelta("fine"))
        renderer.handle(Hostile())  # type: ignore[arg-type]
        renderer.handle(Warning("still working"))
        assert "still working" in buffer.getvalue()

    def test_settled_prose_is_printed_once(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(StepStarted(step=1, max_steps=5, prompt_tokens_estimate=10))
        renderer.handle(TextDelta("The answer is 42."))
        renderer.handle(
            StepFinished(
                step=1,
                message=Message.assistant(TextPart("The answer is 42.")),
                finish_reason="stop",
                usage=Usage(10, 5),
                latency_s=0.5,
            )
        )
        assert buffer.getvalue().count("The answer is 42.") == 1


class TestApprovalPrompt:
    def test_yes_approves(self, monkeypatch, captured) -> None:
        console, _ = captured
        monkeypatch.setattr("builtins.input", lambda: "y")
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="run_bash", risk=RiskLevel.MUTATING, summary="run tests")
        )
        assert decision == Decision(approved=True)

    def test_bare_enter_approves(self, monkeypatch, captured) -> None:
        console, _ = captured
        monkeypatch.setattr("builtins.input", lambda: "")
        assert prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        ).approved

    def test_no_declines(self, monkeypatch, captured) -> None:
        console, _ = captured
        monkeypatch.setattr("builtins.input", lambda: "n")
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        )
        assert not decision.approved and not decision.abort

    def test_always_remembers(self, monkeypatch, captured) -> None:
        console, _ = captured
        monkeypatch.setattr("builtins.input", lambda: "a")
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        )
        assert decision.approved and decision.remember

    def test_always_is_not_offered_for_a_dangerous_call(self, monkeypatch, captured) -> None:
        # Its signature is the command's full text, so remembering it could not
        # be reused anyway — and offering it invites the wrong habit.
        console, buffer = captured
        monkeypatch.setattr("builtins.input", lambda: "n")
        prompt_for_approval(
            console,
            ApprovalRequest(tool="run_bash", risk=RiskLevel.DANGEROUS, summary="rm -rf build"),
        )
        assert "always" not in buffer.getvalue().lower()

    def test_always_is_ignored_for_a_dangerous_call(self, monkeypatch, captured) -> None:
        console, _ = captured
        answers = iter(["a", "n"])
        monkeypatch.setattr("builtins.input", lambda: next(answers))
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.DANGEROUS, summary="s")
        )
        assert not decision.approved

    def test_quit_aborts_the_run(self, monkeypatch, captured) -> None:
        console, _ = captured
        monkeypatch.setattr("builtins.input", lambda: "q")
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        )
        assert decision.abort and not decision.approved

    def test_an_unrecognised_answer_asks_again(self, monkeypatch, captured) -> None:
        console, buffer = captured
        answers = iter(["maybe", "y"])
        monkeypatch.setattr("builtins.input", lambda: next(answers))
        assert prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        ).approved
        assert "y, n, a, or q" in buffer.getvalue()

    def test_the_diff_is_shown_so_the_answer_is_informed(self, monkeypatch, captured) -> None:
        console, buffer = captured
        monkeypatch.setattr("builtins.input", lambda: "y")
        prompt_for_approval(
            console,
            ApprovalRequest(
                tool="edit_file",
                risk=RiskLevel.MUTATING,
                summary="edit app.py (+1/-1)",
                detail="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-was\n+now\n",
            ),
        )
        out = buffer.getvalue()
        assert "-was" in out and "+now" in out

    def test_end_of_input_aborts_rather_than_approving(self, monkeypatch, captured) -> None:
        # A closed stdin must never read as consent.
        console, _ = captured

        def closed() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", closed)
        decision = prompt_for_approval(
            console, ApprovalRequest(tool="t", risk=RiskLevel.MUTATING, summary="s")
        )
        assert not decision.approved and decision.abort


class TestPricing:
    """Rates are supplied by the user; none ship with the project."""

    RATES = {"some-model": Price(1.0, 4.0, 0.1), "some-model-mini": Price(0.1, 0.4)}

    def test_no_rates_means_no_cost_reported(self) -> None:
        # The whole point: a built-in table would go stale, and a stale price
        # states a dollar figure that is simply wrong.
        assert price_for("some-model") is None
        assert estimate_cost("some-model", prompt_tokens=1000, completion_tokens=100) is None

    def test_a_configured_rate_is_found(self) -> None:
        assert price_for("some-model", self.RATES) is not None

    def test_matching_is_by_prefix_so_dated_names_work(self) -> None:
        # Deployment names carry suffixes like "-2024-07-18".
        assert price_for("some-model-2099-01-01", self.RATES) is not None

    def test_the_longest_prefix_wins(self) -> None:
        specific = price_for("some-model-mini-latest", self.RATES)
        general = price_for("some-model-latest", self.RATES)
        assert specific is not None and general is not None
        assert specific.input_per_m < general.input_per_m

    def test_an_unlisted_model_has_no_rate(self) -> None:
        assert price_for("something-else", self.RATES) is None

    def test_cost_grows_with_usage(self) -> None:
        small = estimate_cost(
            "some-model", prompt_tokens=1000, completion_tokens=100, prices=self.RATES
        )
        large = estimate_cost(
            "some-model", prompt_tokens=100_000, completion_tokens=10_000, prices=self.RATES
        )
        assert small is not None and large is not None and large > small

    def test_cached_tokens_are_cheaper(self) -> None:
        fresh = estimate_cost(
            "some-model", prompt_tokens=10_000, completion_tokens=0, prices=self.RATES
        )
        cached = estimate_cost(
            "some-model",
            prompt_tokens=10_000,
            completion_tokens=0,
            cached_tokens=10_000,
            prices=self.RATES,
        )
        assert fresh is not None and cached is not None and cached < fresh

    def test_a_rate_without_a_cached_price_bills_cached_as_fresh(self) -> None:
        rates = {"some-model-mini": Price(0.1, 0.4)}
        plain = estimate_cost(
            "some-model-mini", prompt_tokens=1000, completion_tokens=0, prices=rates
        )
        cached = estimate_cost(
            "some-model-mini",
            prompt_tokens=1000,
            completion_tokens=0,
            cached_tokens=1000,
            prices=rates,
        )
        assert plain == cached

    def test_cached_cannot_exceed_prompt_tokens(self) -> None:
        # Guards against a provider reporting inconsistent numbers.
        cost = estimate_cost(
            "some-model",
            prompt_tokens=100,
            completion_tokens=0,
            cached_tokens=99_999,
            prices=self.RATES,
        )
        assert cost is not None and cost >= 0

    def test_zero_usage_costs_nothing(self) -> None:
        cost = estimate_cost(
            "some-model", prompt_tokens=0, completion_tokens=0, prices=self.RATES
        )
        assert cost == 0.0

    def test_rates_are_parsed_from_configuration(self) -> None:
        table = parse_prices(
            {
                "My-Model": {"input_per_m": 0.5, "output_per_m": 2.0},
                "other": {"input_per_m": 1, "output_per_m": 2, "cached_input_per_m": 0.25},
            }
        )
        assert table["my-model"].output_per_m == 2.0  # names are lowercased
        assert table["other"].cached_input_per_m == 0.25

    def test_a_malformed_rate_is_skipped_not_fatal(self) -> None:
        # A typo in a cosmetic setting must not stop the agent working.
        table = parse_prices(
            {
                "good": {"input_per_m": 1.0, "output_per_m": 2.0},
                "missing-output": {"input_per_m": 1.0},
                "not-a-number": {"input_per_m": "cheap", "output_per_m": 2.0},
                "not-a-table": 3.0,
            }
        )
        assert set(table) == {"good"}


class TestReplay:
    def test_a_recorded_run_is_renarrated(self, tmp_path: Path, capsys) -> None:
        trace = tmp_path / "session.jsonl"
        trace.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "type": "session",
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "workspace": "/proj",
                        "approval_mode": "auto-edit",
                    },
                    {"type": "user", "text": "fix the bug", "t": 0.1},
                    {"type": "step_started", "step": 1, "prompt_tokens_estimate": 1500, "t": 0.2},
                    {
                        "type": "tool_started",
                        "name": "edit_file",
                        "arguments": {"path": "app.py"},
                        "t": 0.5,
                    },
                    {
                        "type": "tool_finished",
                        "name": "edit_file",
                        "is_error": False,
                        "content": "Edited app.py",
                        "t": 0.6,
                    },
                    {
                        "type": "step_finished",
                        "step": 1,
                        "finish_reason": "stop",
                        "latency_s": 1.2,
                        "usage": {"prompt": 1500, "completion": 40},
                        "message": {"parts": [{"type": "text", "text": "Done."}]},
                        "t": 1.8,
                    },
                    {
                        "type": "run_finished",
                        "reason": "finished",
                        "steps": 1,
                        "elapsed_s": 2.0,
                        "usage": {"prompt": 1500, "completion": 40},
                        "t": 2.0,
                    },
                )
            ),
            encoding="utf-8",
        )

        assert main(["--replay", str(trace)]) == 0
        out = capsys.readouterr().out
        assert "fix the bug" in out
        assert "edit_file" in out and "app.py" in out
        assert "Done." in out
        assert "finished" in out

    def test_a_missing_trace_is_an_error(self, tmp_path: Path, capsys) -> None:
        assert main(["--replay", str(tmp_path / "absent.jsonl")]) == 2
        assert "Could not read" in capsys.readouterr().out

    def test_an_empty_trace_is_reported(self, tmp_path: Path, capsys) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert main(["--replay", str(empty)]) == 1
        assert "no events" in capsys.readouterr().out
