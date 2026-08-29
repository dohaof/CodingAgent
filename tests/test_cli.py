"""The command-line surface: parsing, rendering, pricing, and interaction.

Rendering is tested by writing to a captured console rather than a terminal, so
the assertions are about what information reaches the user — the diff, the exit
code, the cost — and not about escape codes.
"""

from __future__ import annotations

import asyncio
import io
import json
import signal
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text
from textual.events import MouseDown
from textual.widgets import Input, TextArea

from cagent.agent.approval import Decision
from cagent.agent.engine import Agent, TurnResult
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
    UserMessage,
    Warning,
)
from cagent.agent.trace import TraceWriter, history_from_trace, read_trace
from cagent.cli.app import (
    _command,
    _install_interrupt_handler,
    _overrides,
    _print_help,
    _repl,
    build_parser,
    main,
)
from cagent.cli.pricing import Price, estimate_cost, parse_prices, price_for
from cagent.cli.render import ConsoleRenderer, _restored_detail, prompt_for_approval
from cagent.cli.tui import CagentTui, ComposerInput, TranscriptTextArea, _ApprovalWaiter
from cagent.config import AgentConfig
from cagent.tools.base import ApprovalRequest, ToolOutcome
from cagent.types import Message, RiskLevel, TextPart, ToolCallPart, ToolResultPart, Usage


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

    def test_trace_cli_flags_are_removed(self) -> None:
        for option in ("--replay", "--resume", "--max-steps"):
            with pytest.raises(SystemExit):
                build_parser().parse_args([option, "trace.jsonl"])

    def test_flags_map_onto_config_fields(self) -> None:
        args = build_parser().parse_args(
            [
                "--base-url", "https://api.example.com/v1",
                "--model", "m",
                "task",
            ]
        )
        overrides = _overrides(args)
        assert overrides["base_url"] == "https://api.example.com/v1"
        assert overrides["model"] == "m"

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

    def test_outside_workspace_flag_enables_unrestricted_paths(self) -> None:
        args = build_parser().parse_args(["--allow-outside-workspace", "task"])
        assert _overrides(args)["allow_outside_workspace"] is True

    def test_docker_sandbox_flags_map_to_config(self) -> None:
        args = build_parser().parse_args(
            [
                "--sandbox", "docker",
                "--sandbox-sync", "ask",
                "--sandbox-image", "local/cagent:latest",
                "--sandbox-memory-mb", "256",
                "--sandbox-cpus", "1.5",
                "--sandbox-pids", "64",
                "--sandbox-workspace-mb", "128",
                "task",
            ]
        )
        overrides = _overrides(args)
        assert overrides["sandbox_mode"] == "docker"
        assert overrides["sandbox_sync"] == "ask"
        assert overrides["sandbox_image"] == "local/cagent:latest"
        assert overrides["sandbox_memory_mb"] == 256
        assert overrides["sandbox_cpus"] == 1.5
        assert overrides["sandbox_pids"] == 64
        assert overrides["sandbox_workspace_mb"] == 128

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
def write_config() -> Callable[..., Path]:
    """Write a ``.cagent.toml`` where the loader will look for it.

    Configuration comes from a file and from flags, so a test that needs a
    settled endpoint writes one. ``isolate_config`` has already pointed both
    home and the working directory at an empty scratch directory, so this lands
    there and never near the developer's own configuration.
    """

    def write(**settings: object) -> Path:
        lines = ["[cagent]"]
        # JSON and TOML agree on strings, numbers, and booleans, which is all
        # these tests set.
        lines += [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
        path = Path.cwd() / ".cagent.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return write


class TestInformationalCommands:
    def test_help_overview_is_concise(self, captured) -> None:
        console, buffer = captured
        _print_help(console)
        out = buffer.getvalue()
        assert "/help <instruct>" in out
        assert "/sandbox" in out
        assert "Docker sandbox" not in out

    def test_help_instruction_shows_detail(self, captured) -> None:
        console, buffer = captured
        _print_help(console, "sandbox")
        out = buffer.getvalue()
        assert "Docker sandbox" in out
        assert "/sandbox apply" in out
        assert "/sandbox rollback" in out
        assert "/sandbox sync never|ask|always" in out

    def test_help_resume_shows_detail(self, captured) -> None:
        console, buffer = captured
        _print_help(console, "resume")
        out = buffer.getvalue()
        assert "/resume" in out and "list saved conversations" in out
        assert "/resume ID" in out and "/resume PATH" in out
        assert "current Agent" in out

    def test_help_undo_explains_context_only_scope(self, captured) -> None:
        console, buffer = captured
        _print_help(console, "undo")
        out = buffer.getvalue()
        assert "/undo" in out
        assert "Tool calls and results are removed as one unit" in out
        assert "does not revert files" in out

    def test_undo_checkpoints_the_remaining_history(
        self, captured, config: AgentConfig
    ) -> None:
        console, buffer = captured
        config.trace_dir = config.workspace / "traces"
        writer = TraceWriter.create(config, session_id="undo-command")
        assert writer is not None
        writer.handle(UserMessage("stale trace history"))

        class StubAgent:
            def __init__(self) -> None:
                self.context = SimpleNamespace(
                    history=[
                        Message.user("keep request"),
                        Message.assistant(TextPart("keep answer")),
                        Message.user("remove request"),
                        Message.assistant(TextPart("remove answer")),
                    ]
                )
                self.guard = SimpleNamespace(note_progress=lambda: None)
                self.sink = SimpleNamespace(sinks=[writer])

            def undo_last_turn(self) -> int:
                return Agent.undo_last_turn(self)  # type: ignore[arg-type]

        agent = StubAgent()
        _command(console, agent, config, "/undo")  # type: ignore[arg-type]
        writer.close()

        restored = history_from_trace(read_trace(writer.path))
        assert [message.text for message in restored] == ["keep request", "keep answer"]
        assert "command effects were not reverted" in buffer.getvalue()

    def test_trace_instruction_is_not_exposed(self, captured, tmp_path: Path) -> None:
        console, buffer = captured
        _print_help(console)
        assert "/trace" not in buffer.getvalue()

        buffer.seek(0)
        buffer.truncate(0)
        _command(console, object(), AgentConfig(workspace=tmp_path), "/trace")  # type: ignore[arg-type]
        assert "unknown command /trace" in buffer.getvalue()

    def test_resume_command_restores_trace_into_current_agent(
        self, captured, tmp_path: Path
    ) -> None:
        console, buffer = captured
        trace = tmp_path / "session.jsonl"
        trace.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"type": "session", "workspace": str(tmp_path)},
                    {"type": "user", "text": "old task"},
                    {
                        "type": "step_finished",
                        "message": {
                            "role": "assistant",
                            "parts": [{"type": "text", "text": "old answer"}],
                        },
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )

        class StubAgent:
            def __init__(self) -> None:
                self.restored = []

            def restore_history(self, messages) -> None:
                self.restored = messages

        agent = StubAgent()
        config = AgentConfig(workspace=tmp_path)

        _command(console, agent, config, f"/resume {trace}")  # type: ignore[arg-type]

        assert [message.text for message in agent.restored] == ["old task", "old answer"]
        out = buffer.getvalue()
        assert "Restored context" in out and "old task" in out and "old answer" in out
        assert "Live conversation continues from here" in out
        assert "resumed 2 message(s)" in out

    def test_resume_without_argument_presents_recent_traces(
        self, captured, tmp_path: Path, monkeypatch
    ) -> None:
        console, buffer = captured
        trace_dir = tmp_path / ".cagent" / "traces"
        trace_dir.mkdir(parents=True)
        for session_id, prompt in (("new-session", "new task"), ("old-session", "old task")):
            (trace_dir / f"{session_id}.jsonl").write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"type": "session", "workspace": str(tmp_path)},
                        {"type": "user", "text": prompt},
                        {
                            "type": "step_finished",
                            "step": 1,
                            "message": {
                                "role": "assistant",
                                "parts": [{"type": "text", "text": "old answer"}],
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
        # Ensure the first directory entry is the newest one deterministically.
        now = __import__("time").time()
        (trace_dir / "new-session.jsonl").touch()
        (trace_dir / "old-session.jsonl").touch()
        import os

        os.utime(trace_dir / "new-session.jsonl", (now + 2, now + 2))
        os.utime(trace_dir / "old-session.jsonl", (now, now))

        class StubAgent:
            def __init__(self) -> None:
                self.restored = []

            def restore_history(self, messages) -> None:
                self.restored = messages

        agent = StubAgent()
        config = AgentConfig(workspace=tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "1")

        _command(console, agent, config, "/resume")  # type: ignore[arg-type]

        assert [message.text for message in agent.restored] == ["new task", "old answer"]
        out = buffer.getvalue()
        assert "Saved conversations" in out and "new task" in out
        assert "resumed 2 message(s)" in out

    def test_resume_short_id_is_found_in_the_default_trace_directory(
        self, captured, tmp_path: Path
    ) -> None:
        console, buffer = captured
        trace_dir = tmp_path / ".cagent" / "traces"
        trace_dir.mkdir(parents=True)
        path = trace_dir / "f52a54aac599.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"type": "session", "workspace": str(tmp_path)},
                    {"type": "user", "text": "continue this"},
                )
            )
            + "\n",
            encoding="utf-8",
        )

        class StubAgent:
            def __init__(self) -> None:
                self.restored = []

            def restore_history(self, messages) -> None:
                self.restored = messages

        agent = StubAgent()
        config = AgentConfig(workspace=tmp_path)
        _command(console, agent, config, "/resume f52a54aac599")  # type: ignore[arg-type]

        assert [message.text for message in agent.restored] == ["continue this"]
        assert "resumed 1 message(s)" in buffer.getvalue()

    def test_resume_explains_when_trace_files_are_empty(
        self, captured, tmp_path: Path
    ) -> None:
        console, buffer = captured
        trace_dir = tmp_path / ".cagent" / "traces"
        trace_dir.mkdir(parents=True)
        (trace_dir / "empty.jsonl").write_text(
            '{"type":"session"}\n{"type":"run_finished","steps":0}\n',
            encoding="utf-8",
        )
        config = AgentConfig(workspace=tmp_path)

        class StubAgent:
            def restore_history(self, messages) -> None:
                raise AssertionError("empty traces must not be restored")

        _command(console, StubAgent(), config, "/resume")  # type: ignore[arg-type]
        out = buffer.getvalue()
        assert "found 1 trace file" in out
        assert "non-empty user turn" in out

    def test_resume_uses_a_relative_configured_trace_directory(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace=tmp_path, trace_dir=Path("saved-traces"))
        assert config.trace_dir == (tmp_path / "saved-traces").resolve()

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
        self, capsys, tmp_path, write_config
    ) -> None:
        write_config(
            api_key="sk-secret",
            base_url="https://api.example.com/v1",
            model="file-model",
        )
        assert main(["--show-config", "--workspace", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "https://api.example.com/v1" in out
        assert "file-model" in out
        assert "set" in out

    def test_show_config_reports_unrestricted_path_access(self, capsys, tmp_path) -> None:
        assert (
            main(
                [
                    "--show-config",
                    "--workspace",
                    str(tmp_path),
                    "--allow-outside-workspace",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "path boundary" in out
        assert "unrestricted" in out
        assert "shell execution" in out
        assert "host (unrestricted)" in out

    def test_show_config_reports_automatic_sandbox_policy_by_default(
        self, capsys, tmp_path
    ) -> None:
        assert main(["--show-config", "--workspace", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "path boundary" in out and "workspace-only" in out
        assert "shell execution" in out and "host (unrestricted)" in out and "auto" in out
        assert "WARNING" in out

    def test_explicit_workspace_loads_its_project_config(self, capsys, tmp_path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (project / ".cagent.toml").write_text(
            '[cagent]\nbase_url = "https://project.example/v1"\nmodel = "project-model"\n',
            encoding="utf-8",
        )

        assert main(["--show-config", "--workspace", str(project)]) == 0
        out = capsys.readouterr().out
        assert "https://project.example/v1" in out
        assert "project-model" in out

    def test_show_config_never_prints_the_key(self, capsys, tmp_path, write_config) -> None:
        write_config(api_key="sk-do-not-print-me")
        main(["--show-config", "--workspace", str(tmp_path)])
        assert "sk-do-not-print-me" not in capsys.readouterr().out

    def test_show_config_works_when_nothing_is_configured(self, capsys, tmp_path) -> None:
        # This command is most useful precisely when the configuration is
        # incomplete, so it reports gaps rather than raising.
        assert main(["--show-config", "--workspace", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "not set" in out
        assert ".cagent.toml" in out

    def test_show_config_names_the_full_request_url(self, capsys, tmp_path) -> None:
        main(
            [
                "--show-config",
                "--base-url", "https://api.example.com/v1/",
                "--model", "m",
                "--workspace", str(tmp_path),
            ]
        )
        assert "https://api.example.com/v1/chat/completions" in capsys.readouterr().out

    def test_show_config_names_the_anthropic_path(self, capsys, tmp_path) -> None:
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
        self, capsys, tmp_path, write_config
    ) -> None:
        write_config(api_key="sk-gateway")
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

    def test_a_missing_endpoint_is_reported_before_any_request(self, capsys, tmp_path) -> None:
        code = main(["--model", "m", "--workspace", str(tmp_path), "do a thing"])
        out = capsys.readouterr().out
        assert code == 2
        assert "base_url" in out.lower()

    def test_a_missing_model_is_reported_rather_than_guessed(self, capsys, tmp_path) -> None:
        # There is deliberately no default model: a name frozen at release time
        # fails later as an unhelpful remote 404.
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

    def test_a_missing_key_names_the_endpoint(self, capsys, tmp_path) -> None:
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

    def test_the_missing_key_message_points_at_the_config_file(self, capsys, tmp_path) -> None:
        # The key has exactly one home now, so the message names it — and names
        # the endpoint, rather than sending someone to a vendor variable that
        # has nothing to do with their self-hosted gateway.
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
        assert ".cagent.toml" in out
        assert "DEEPSEEK_API_KEY" not in out

    def test_the_hint_shows_a_complete_configuration(self, capsys, tmp_path) -> None:
        main(["--workspace", str(tmp_path), "do a thing"])
        out = capsys.readouterr().out
        for expected in ("[cagent]", "base_url", "model", "api_key"):
            assert expected in out

    def test_no_key_permits_a_local_server(self, capsys, tmp_path) -> None:
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

    def test_the_environment_is_not_a_configuration_layer(
        self, capsys, monkeypatch, tmp_path, write_config
    ) -> None:
        # Every setting has one spelling, in the file. A CAGENT_* variable named
        # after a field is not a second way to set it, and neither is a vendor
        # key variable that happens to be exported.
        monkeypatch.setenv("CAGENT_MODEL", "env-model")
        monkeypatch.setenv("CAGENT_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-vendor-var")
        write_config(base_url="https://gw.example.com/v1", model="file-model")
        code = main(["--show-config", "--workspace", str(tmp_path)])
        out = capsys.readouterr().out
        assert code == 0
        assert "file-model" in out
        assert "env-model" not in out
        assert "not set" in out  # the key, which only the file could have set


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

    def test_numbered_tool_output_keeps_the_first_line_aligned(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer._print_result_body(
            '     1\t"""Basic arithmetic helpers."""\n     2\t\n     3\tdef add(a, b):',
            style="dim",
        )
        lines = buffer.getvalue().splitlines()
        assert lines[0].startswith("         1")
        assert lines[1].startswith("         2")
        assert lines[2].startswith("         3")

    def test_restored_numbered_detail_keeps_the_first_line_aligned(self) -> None:
        rendered = _restored_detail("     1\tfirst\n     2\tsecond")
        assert [item.plain for item in rendered] == [
            "         1\tfirst",
            "         2\tsecond",
        ]

    def test_a_step_reports_its_share_of_the_window(self, config, captured) -> None:
        console, buffer = captured
        renderer = ConsoleRenderer(config, console=console)
        renderer.handle(StepStarted(step=3, prompt_tokens_estimate=64_000))
        out = buffer.getvalue()
        assert "step 3" in out and "step 3/" not in out and "64,000" in out and "50%" in out

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
        renderer.handle(StepStarted(step=1, prompt_tokens_estimate=100))
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
        renderer.handle(StepStarted(step=1, prompt_tokens_estimate=10))
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


class TestInterruptHandling:
    def test_ctrl_c_interrupts_a_busy_line_turn_but_idle_ctrl_c_exits(
        self, captured, monkeypatch
    ) -> None:
        console, _ = captured

        class StubAgent:
            def __init__(self) -> None:
                self.interruptions = 0
                self.abort = SimpleNamespace(is_set=lambda: False)

            def interrupt(self) -> None:
                self.interruptions += 1

            def reset_interrupt(self) -> None:
                pass

            def run_turn(self, text: str) -> TurnResult:
                assert text == "first task"
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)  # type: ignore[call-arg]
                return TurnResult("", 1, Usage(1, 1), "aborted")

        agent = StubAgent()
        previous = signal.getsignal(signal.SIGINT)
        try:
            state = _install_interrupt_handler(agent, console)
            commands = iter(("first task", "/exit"))
            monkeypatch.setattr("builtins.input", lambda: next(commands))
            assert _repl(console, agent, SimpleNamespace(), state) == 0
            assert agent.interruptions == 1
            assert not state.busy

            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)  # type: ignore[call-arg]
        finally:
            signal.signal(signal.SIGINT, previous)


class TestTui:
    def test_undo_rebuilds_the_transcript_without_the_latest_user_turn(
        self, config: AgentConfig
    ) -> None:
        app = CagentTui(config)

        async def exercise_undo() -> None:
            async with app.run_test(size=(100, 30)):
                app.agent.context.history = [
                    Message.user("keep this request"),
                    Message.assistant(TextPart("keep this answer")),
                    Message.user("remove this request"),
                    Message.assistant(
                        ToolCallPart("call", "read_file", {"path": "a.py"})
                    ),
                    Message.from_tool_results([ToolResultPart("call", "contents")]),
                    Message.assistant(TextPart("remove this answer")),
                ]
                app._handle_command("/undo")

                history = app.agent.context.history
                transcript = app.query_one("#conversation", TextArea).text
                assert [message.text for message in history] == [
                    "keep this request",
                    "keep this answer",
                ]
                assert "keep this request" in transcript
                assert "remove this request" not in transcript
                assert "command effects were not reverted" in transcript.lower()

        try:
            asyncio.run(exercise_undo())
        finally:
            app.emergency_close()

    def test_full_screen_app_mounts_headlessly(self, config: AgentConfig) -> None:
        class SmokeApp(CagentTui):
            mounted_ok = False

            def on_mount(self) -> None:
                super().on_mount()
                self.mounted_ok = bool(self.query_one("#conversation")) and bool(
                    self.query_one("#composer")
                )
                self.exit(result=7)

        app = SmokeApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 7
            assert app.mounted_ok
        finally:
            app.emergency_close()

    def test_startup_status_is_rendered_as_selectable_transcript_panel(
        self, config: AgentConfig
    ) -> None:
        class StartupApp(CagentTui):
            def on_mount(self) -> None:
                super().on_mount()
                self._render_event(
                    RunStarted(
                        task="(interactive)",
                        model="test-model",
                        endpoint="https://api.test.invalid/v1",
                        system_tokens=1200,
                        tool_names=("read_file", "edit_file"),
                    )
                )
                transcript = self.query_one("#conversation", TextArea)
                assert "test-model" in transcript.text
                assert "https://api.test.invalid/v1" in transcript.text
                assert "WORKSPACE" in transcript.text
                assert "PATHS" in transcript.text
                assert "SHELL" in transcript.text
                assert "host (unrestricted)" in transcript.text
                assert "read_file" in transcript.text
                panel_lines = transcript.text.splitlines()
                assert len({len(line) for line in panel_lines}) == 1
                self.exit(result=0)

        app = StartupApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
        finally:
            app.emergency_close()

    def test_startup_panel_right_border_stays_in_one_visual_column(
        self, config: AgentConfig
    ) -> None:
        app = CagentTui(config)

        async def inspect_rendered_panel() -> None:
            async with app.run_test(size=(120, 30)) as pilot:
                transcript = app.query_one("#conversation", TranscriptTextArea)
                transcript.load_text("")
                app._render_event(
                    RunStarted(
                        task="(interactive)",
                        model="test-model",
                        endpoint="https://api.test.invalid/v1",
                        system_tokens=1200,
                        tool_names=("read_file", "edit_file"),
                    )
                )
                await pilot.pause()

                source_lines = transcript.text.splitlines()
                rendered_lines = [
                    "".join(segment.text for segment in transcript.render_line(index)._segments)
                    for index in range(len(source_lines))
                ]
                right_borders = [
                    max(
                        index
                        for index, character in enumerate(line)
                        if character in {"┐", "│", "┘"}
                    )
                    for line in rendered_lines
                ]

                assert len(set(right_borders)) == 1
                assert any("read_file" in line for line in rendered_lines)

        try:
            asyncio.run(inspect_rendered_panel())
        finally:
            app.emergency_close()

    def test_transcript_preserves_semantic_colors_for_roles_tools_and_diffs(self) -> None:
        transcript = TranscriptTextArea(
            "USER\nASSISTANT\nTOOL: EDIT_FILE(path)\n--- a/app.py\n+++ b/app.py\n-old\n+new",
            read_only=True,
        )
        assert str(transcript.get_line(0).spans[0].style) == "bold cyan reverse"
        assert str(transcript.get_line(1).spans[0].style) == "bold green reverse"
        assert str(transcript.get_line(2).spans[0].style) == "bold bright_yellow"
        assert str(transcript.get_line(3).spans[0].style) == "red"
        assert str(transcript.get_line(4).spans[0].style) == "green"
        assert str(transcript.get_line(5).spans[0].style) == "red"
        assert str(transcript.get_line(6).spans[0].style) == "green"

    def test_transcript_renders_markdown_while_retaining_selectable_source(self) -> None:
        transcript = TranscriptTextArea(
            "**bold**\n*italic*\n***both***\n~~strike~~\n# heading\n```python\nprint(1)\n```",
            read_only=True,
        )

        assert transcript.document.get_line(0) == "**bold**"
        assert transcript.document.get_line(1) == "*italic*"
        assert transcript.document.get_line(2) == "***both***"
        assert transcript.document.get_line(3) == "~~strike~~"

        prose = transcript.get_line(0)
        assert any(str(span.style) == "bold" for span in prose.spans)
        assert any(str(span.style) == "italic" for span in transcript.get_line(1).spans)
        assert any(
            "bold" in str(span.style) and "italic" in str(span.style)
            for span in transcript.get_line(2).spans
        )
        assert any("strike" in str(span.style) for span in transcript.get_line(3).spans)
        assert any("underline" in str(span.style) for span in transcript.get_line(4).spans)

        visual = prose.copy()
        TranscriptTextArea._replace_markdown_markers(visual)
        assert visual.plain.replace("\u200b", "") == "bold"
        assert "\u200b" in visual.plain
        assert str(transcript.get_line(6).spans[0].style) == "cyan"

    def test_tool_events_are_written_with_distinguishable_transcript_labels(
        self, config: AgentConfig
    ) -> None:
        class ToolApp(CagentTui):
            def on_mount(self) -> None:
                super().on_mount()
                self._render_event(
                    ToolStarted(
                        call=ToolCallPart(
                            id="call-1", name="read_file", arguments={"path": "add.py"}
                        ),
                        risk=RiskLevel.SAFE,
                    )
                )
                conversation = self.query_one("#conversation", TextArea)
                assert "TOOL: READ_FILE" in conversation.text
                assert str(conversation.get_line(0).spans[0].style) == "bold bright_yellow"
                self._render_event(
                    ToolFinished(
                        call=ToolCallPart(
                            id="call-1", name="read_file", arguments={"path": "add.py"}
                        ),
                        outcome=ToolOutcome.ok(
                            "     1\tcontent",
                            metadata={"path": "add.py", "lines_shown": 1, "total_lines": 1},
                        ),
                        duration_s=0.01,
                    )
                )
                assert "TOOL RESULT: 1 of 1 lines" in conversation.text
                assert conversation.text.count("add.py") == 1
                assert "\n    1  content" in conversation.text
                assert str(conversation.get_line(1).spans[0].style) == "bold bright_green"
                self.exit(result=0)

        app = ToolApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
        finally:
            app.emergency_close()

    def test_approval_is_inline_and_uses_y_n_always_quit(self, config: AgentConfig) -> None:
        class ApprovalApp(CagentTui):
            waiter = _ApprovalWaiter()

            def on_mount(self) -> None:
                super().on_mount()
                request = ApprovalRequest(
                    tool="edit_file",
                    risk=RiskLevel.MUTATING,
                    summary="update add.py",
                    detail="--- a/add.py\n+++ b/add.py\n-old\n+new",
                )
                self._show_approval(request, self.waiter)
                transcript = self.query_one("#conversation", TextArea)
                assert "APPROVAL · MUTATING · EDIT_FILE" in transcript.text
                assert "[Y]ES  [N]O  [A]LWAYS  [Q]UIT" in transcript.text
                assert not self.query("ApprovalScreen")
                self._handle_approval_input("always")
                assert self.waiter.ready.is_set()
                assert self.waiter.decision == Decision(True, remember=True)
                self.exit(result=0)

        app = ApprovalApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
        finally:
            app.emergency_close()

    def test_shell_approval_separates_purpose_command_and_directory(
        self, config: AgentConfig
    ) -> None:
        class ShellApprovalApp(CagentTui):
            waiter = _ApprovalWaiter()

            def on_mount(self) -> None:
                super().on_mount()
                request = ApprovalRequest(
                    tool="run_bash",
                    risk=RiskLevel.MUTATING,
                    summary="run: python -m pytest — Run the focused tests",
                    detail="$ python -m pytest\nin .",
                )
                self._show_approval(request, self.waiter)
                transcript = self.query_one("#conversation", TextArea).text
                assert "PURPOSE\n  Run the focused tests" in transcript
                assert "COMMAND\n  $ python -m pytest" in transcript
                assert "WORKING DIRECTORY\n  ." in transcript
                assert transcript.count("python -m pytest") == 1
                self._handle_approval_input("n")
                self.exit(result=0)

        app = ShellApprovalApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
        finally:
            app.emergency_close()

    def test_streaming_deltas_stay_in_the_main_conversation(self, config: AgentConfig) -> None:
        class StreamApp(CagentTui):
            def on_mount(self) -> None:
                super().on_mount()
                self._render_event(TextDelta("partial "))
                self._render_event(TextDelta("answer"))
                conversation = self.query_one("#conversation", TextArea)
                assert "partial answer" in conversation.text
                self._render_event(
                    StepFinished(
                        step=1,
                        message=Message.assistant(TextPart("partial answer")),
                        finish_reason="stop",
                        usage=Usage(10, 5),
                        latency_s=0.1,
                    )
                )
                assert conversation.text.count("partial answer") == 1
                assert not self.query("#stream")
                self.exit(result=0)

        app = StreamApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
        finally:
            app.emergency_close()

    def test_search_results_group_repeated_paths(self) -> None:
        grouped = CagentTui._group_search_results(
            "add.py:18- old\nadd.py:19: def shortest():\nother.py:2: call()"
        )
        assert grouped == (
            "add.py\n    18- old\n    19: def shortest():\n\n"
            "other.py\n    2: call()"
        )

    def test_conversation_is_selectable_and_ctrl_c_copies_selection(
        self, config: AgentConfig
    ) -> None:
        class CopyApp(CagentTui):
            copied = ""
            has_topbar = True
            read_only = False

            def on_mount(self) -> None:
                super().on_mount()
                conversation = self.query_one("#conversation", TextArea)
                self._write(Text("selectable transcript"))
                self.screen.set_focus(None)
                conversation.select_all()
                self.action_interrupt()
                self.copied = self.clipboard
                self.has_topbar = bool(self.query("#topbar"))
                self.read_only = conversation.read_only
                self.exit(result=0)

        app = CopyApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
            assert "selectable transcript" in app.copied
            assert app.read_only
            assert not app.has_topbar
        finally:
            app.emergency_close()

    def test_ctrl_c_interrupts_busy_turn_without_closing_session(
        self, config: AgentConfig
    ) -> None:
        app = CagentTui(config)

        async def exercise_interrupt() -> None:
            async with app.run_test(size=(100, 30)):
                app._set_busy(True, "Working")
                app.action_interrupt()
                assert app.agent.abort.is_set()
                assert not app._closing_session
                assert not app._exit_after_turn

        try:
            asyncio.run(exercise_interrupt())
        finally:
            app.emergency_close()

    def test_clicking_and_selecting_transcript_clears_keyboard_input_focus(
        self, config: AgentConfig
    ) -> None:
        app = CagentTui(config)

        async def exercise_transcript() -> None:
            async with app.run_test(size=(100, 30)) as pilot:
                conversation = app.query_one("#conversation", TranscriptTextArea)
                composer = app.query_one("#composer", Input)
                app._write(Text("selectable transcript"))

                assert not conversation.focusable
                assert app.focused is composer
                input_cursor_position = app.cursor_position
                await pilot.mouse_down(conversation, offset=(3, 1))
                await pilot.hover(conversation, offset=(12, 1))
                await pilot.mouse_up(conversation, offset=(12, 1))

                assert app.focused is None
                assert app.cursor_position == input_cursor_position
                assert not conversation._cursor_visible
                assert conversation.selected_text
                await pilot.press("x")
                assert composer.value == ""

        try:
            asyncio.run(exercise_transcript())
        finally:
            app.emergency_close()

    def test_composer_hides_prompt_while_focused_for_ime_preedit(self, config: AgentConfig) -> None:
        app = CagentTui(config)

        async def exercise_composer() -> None:
            async with app.run_test(size=(100, 30)) as pilot:
                composer = app.query_one("#composer", ComposerInput)
                await pilot.pause()
                assert app.focused is composer
                assert composer.placeholder == ""

                composer.set_prompt_hint("Approval: y/n/a/q (Enter = yes)")
                assert composer.placeholder == ""
                app.screen.set_focus(None)
                await pilot.pause()
                assert composer.placeholder == "Approval: y/n/a/q (Enter = yes)"

        try:
            asyncio.run(exercise_composer())
        finally:
            app.emergency_close()

    def test_right_click_copies_selected_transcript(self, config: AgentConfig) -> None:
        app = CagentTui(config)

        async def exercise_context_copy() -> None:
            async with app.run_test(size=(100, 30)):
                conversation = app.query_one("#conversation", TranscriptTextArea)
                app._write(Text("copy from context menu"))
                conversation.select_all()
                event = MouseDown(
                    conversation,
                    x=3,
                    y=1,
                    delta_x=0,
                    delta_y=0,
                    button=3,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
                await conversation._on_mouse_down(event)
                assert "copy from context menu" in app.clipboard

        try:
            asyncio.run(exercise_context_copy())
        finally:
            app.emergency_close()

    def test_resume_replaces_tui_history_and_rebuilds_the_transcript(
        self, config: AgentConfig
    ) -> None:
        trace_dir = config.workspace / ".cagent" / "traces"
        trace_dir.mkdir(parents=True)
        trace = trace_dir / "restored-session.jsonl"
        trace.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"type": "session", "workspace": str(config.workspace)},
                    {"type": "user", "text": "old request"},
                    {
                        "type": "step_finished",
                        "message": {
                            "role": "assistant",
                            "parts": [{"type": "text", "text": "old answer"}],
                        },
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        config.trace_dir = trace_dir

        class ResumeApp(CagentTui):
            rendered = []

            def on_mount(self) -> None:
                super().on_mount()
                self._restore_path(trace)
                self.exit(result=0)

            def _write(self, renderable) -> None:
                self.rendered.append(renderable)
                super()._write(renderable)

        app = ResumeApp(config)
        try:
            assert app.run(headless=True, size=(100, 30)) == 0
            assert [message.text for message in app.agent.context.history] == [
                "old request",
                "old answer",
            ]
            assert app._restored_from == "restored-session"
            assert len(app.rendered) >= 4
        finally:
            app.emergency_close()


class TestApprovalPrompt:
    def test_shortcuts_are_visible_in_approval_menu(self, monkeypatch, captured) -> None:
        console, buffer = captured
        monkeypatch.setattr("builtins.input", lambda: "n")
        prompt_for_approval(
            console, ApprovalRequest(tool="run_bash", risk=RiskLevel.MUTATING, summary="run tests")
        )
        out = buffer.getvalue()
        assert "[y]es" in out and "[n]o" in out and "[a]lways" in out and "[q]uit" in out

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
