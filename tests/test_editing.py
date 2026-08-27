"""The matching ladder and the edit tools.

The ladder exists because a model's ``old_string`` is a copy of what it read,
and copies drift: indentation gets normalised, tabs become spaces, a character
is mistyped. These tests are the specification of how much drift is tolerated
and — just as important — where tolerance stops, because a fuzzy match applied
to the wrong block silently corrupts a file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cagent.tools.diffs import diff_stats, snippet_around, unified_diff
from cagent.tools.edit import (
    EditFileParams,
    EditFileTool,
    MultiEditTool,
    read_text_preserving,
)
from cagent.tools.matching import (
    best_rejected,
    describe_ambiguity,
    find_block,
    replace_block,
)

SOURCE = """\
class Widget:
    def render(self):
        if self.visible:
            return self.template
        return ""
"""


class TestFindBlock:
    def test_exact_match_wins(self) -> None:
        matches = find_block(SOURCE, "        if self.visible:", fuzzy_threshold=0.8)
        assert [m.strategy for m in matches] == ["exact"]
        assert matches[0].similarity == 1.0

    def test_exact_match_offsets_span_the_needle(self) -> None:
        needle = "return self.template"
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        assert SOURCE[match.start : match.end] == needle

    def test_every_exact_occurrence_is_returned(self) -> None:
        text = "a = 1\nb = 2\na = 1\n"
        matches = find_block(text, "a = 1", fuzzy_threshold=0.8)
        assert len(matches) == 2

    def test_uniform_indent_drift_matches_at_the_whitespace_level(self) -> None:
        # The classic failure: the model reproduced the block with its own
        # indentation rather than the file's.
        needle = "def render(self):\n    if self.visible:"
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        assert match.strategy == "whitespace"
        assert "def render(self):" in match.matched_text

    def test_tabs_match_spaces(self) -> None:
        tabbed = "class A:\n\tdef go(self):\n\t\treturn 1\n"
        (match,) = find_block(tabbed, "    def go(self):\n        return 1", fuzzy_threshold=0.8)
        assert match.strategy == "whitespace"

    def test_trailing_whitespace_is_ignored(self) -> None:
        text = "x = 1   \ny = 2\n"
        (match,) = find_block(text, "x = 1\ny = 2", fuzzy_threshold=0.8)
        assert match.strategy == "whitespace"

    def test_whitespace_scores_below_exact_but_above_fuzzy(self) -> None:
        needle = "def render(self):\n    if self.visible:"
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        assert 0.9 < match.similarity < 1.0

    def test_small_typo_matches_fuzzily(self) -> None:
        needle = "        if self.visable:"  # transposed letters
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        assert match.strategy == "fuzzy"
        assert "self.visible" in match.matched_text

    def test_a_needle_too_different_does_not_match(self) -> None:
        assert find_block(SOURCE, "def unrelated_function(x, y, z):", fuzzy_threshold=0.86) == []

    def test_threshold_is_respected(self) -> None:
        needle = "        if self.thing_entirely_different:"
        assert find_block(SOURCE, needle, fuzzy_threshold=0.99) == []

    def test_levels_are_never_mixed(self) -> None:
        # If an exact match exists, a fuzzy near-match elsewhere must not join
        # it and turn a unique edit into an ambiguous one.
        text = "value = 1\nvalue = 2\n"
        matches = find_block(text, "value = 1", fuzzy_threshold=0.5)
        assert len(matches) == 1 and matches[0].strategy == "exact"

    def test_empty_needle_matches_nothing(self) -> None:
        assert find_block(SOURCE, "", fuzzy_threshold=0.8) == []

    def test_ambiguous_fuzzy_candidates_are_all_returned(self) -> None:
        text = "def handler_one(request):\n    pass\n\ndef handler_two(request):\n    pass\n"
        matches = find_block(text, "def handler_xxx(request):\n    pass", fuzzy_threshold=0.7)
        assert len(matches) >= 2

    def test_overlapping_candidates_are_suppressed(self) -> None:
        text = "\n".join(f"line {n}" for n in range(20))
        matches = find_block(text, "line 5\nline 6", fuzzy_threshold=0.6)
        for first, second in zip(matches, matches[1:], strict=False):
            assert first.end <= second.start or second.end <= first.start


class TestBestRejected:
    def test_reports_the_nearest_miss(self) -> None:
        # This is what turns "not found" into a message the model can act on.
        needle = "        if self.thing_entirely_different:"
        rejected = best_rejected(SOURCE, needle, fuzzy_threshold=0.99)
        assert rejected is not None
        assert rejected.similarity < 0.99
        assert "visible" in rejected.matched_text

    def test_returns_nothing_when_the_file_is_unrelated(self) -> None:
        rejected = best_rejected(
            "a\nb\n", "completely unrelated content here", fuzzy_threshold=0.9
        )
        assert rejected is None


class TestReplaceBlock:
    def test_exact_replacement_splices_in_place(self) -> None:
        (match,) = find_block(SOURCE, 'return ""', fuzzy_threshold=0.8)
        assert 'return None' in replace_block(SOURCE, match, "return None")

    def test_replacement_is_reindented_to_the_matched_block(self) -> None:
        # Without this the new code lands at the needle's indentation and the
        # file no longer parses — the reason naive fuzzy replace is unusable.
        needle = "def render(self):\n    if self.visible:"
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        result = replace_block(SOURCE, match, "def render(self):\n    if self.shown:")
        assert "    def render(self):" in result
        assert "        if self.shown:" in result

    def test_blank_lines_are_not_indented(self) -> None:
        needle = "def render(self):\n    if self.visible:"
        (match,) = find_block(SOURCE, needle, fuzzy_threshold=0.8)
        result = replace_block(SOURCE, match, "def render(self):\n\n    if self.shown:")
        assert "\n\n" in result and "    \n" not in result

    def test_line_count_is_preserved_around_the_splice(self) -> None:
        (match,) = find_block(SOURCE, "        return self.template", fuzzy_threshold=0.8)
        result = replace_block(SOURCE, match, "        return self.tpl")
        assert result.count("\n") == SOURCE.count("\n")


class TestDescribeAmbiguity:
    def test_lists_line_numbers_and_excerpts(self) -> None:
        text = "value = 1\nfiller\nvalue = 1\n"
        matches = find_block(text, "value = 1", fuzzy_threshold=0.8)
        message = describe_ambiguity(matches, text)
        assert "line 1" in message and "line 3" in message
        assert "replace_all" in message


class TestDiffs:
    def test_unified_diff_has_headers_and_changes(self) -> None:
        diff = unified_diff("a\nb\n", "a\nc\n", "f.py")
        assert diff.startswith("--- a/f.py")
        assert "+++ b/f.py" in diff
        assert "-b" in diff and "+c" in diff

    def test_identical_text_produces_no_diff(self) -> None:
        assert unified_diff("same\n", "same\n", "f.py").strip() == ""

    def test_diff_stats_counts_changes_not_headers(self) -> None:
        diff = unified_diff("a\nb\nc\n", "a\nx\ny\nc\n", "f.py")
        added, removed = diff_stats(diff)
        assert (added, removed) == (2, 1)

    def test_missing_trailing_newline_is_marked(self) -> None:
        diff = unified_diff("a\nb", "a\nc", "f.py")
        assert "No newline at end of file" in diff

    def test_snippet_is_line_numbered_with_context(self) -> None:
        text = "\n".join(f"line {n}" for n in range(1, 21))
        start = text.index("line 10")
        snippet = snippet_around(text, start, start + 7, context_lines=2)
        assert "    10\tline 10" in snippet
        assert "line 8" in snippet and "line 12" in snippet
        assert "line 5" not in snippet


def write(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> None:
    path.write_bytes(text.replace("\n", newline).encode(encoding))


class TestEditFileTool:
    def test_exact_edit_applies_and_reports_a_diff(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "def add(a, b):\n    return a - b\n")

        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "return a - b", "new_string": "return a + b"},
            harness.ctx,
        )

        assert not outcome.is_error, outcome.content
        assert target.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
        assert outcome.display is not None and "+    return a + b" in outcome.display
        assert outcome.metadata["strategy"] == "exact"

    def test_result_shows_the_post_edit_lines(self, make_ctx, tmp_path: Path) -> None:
        # So the model can see what it produced without spending a read_file.
        harness = make_ctx()
        write(tmp_path / "app.py", "x = 1\ny = 2\nz = 3\n")
        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "y = 2", "new_string": "y = 22"}, harness.ctx
        )
        assert "y = 22" in outcome.content

    def test_fuzzy_edit_records_its_similarity(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        write(tmp_path / "app.py", SOURCE)
        outcome = EditFileTool().invoke(
            {
                "path": "app.py",
                "old_string": "        if self.visable:",
                "new_string": "        if self.shown:",
            },
            harness.ctx,
        )
        assert not outcome.is_error, outcome.content
        assert outcome.metadata["strategy"] == "fuzzy"
        assert 0.86 <= float(outcome.metadata["similarity"]) < 1.0  # type: ignore[arg-type]

    def test_ambiguous_match_is_refused_with_line_numbers(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "value = 1\nfiller\nvalue = 1\n")

        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "value = 1", "new_string": "value = 2"}, harness.ctx
        )

        assert outcome.is_error
        assert "line 1" in outcome.content and "line 3" in outcome.content
        assert target.read_text(encoding="utf-8") == "value = 1\nfiller\nvalue = 1\n"

    def test_no_match_names_the_closest_candidate(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        write(tmp_path / "app.py", SOURCE)
        outcome = EditFileTool().invoke(
            {
                "path": "app.py",
                "old_string": "        if self.thing_entirely_different:",
                "new_string": "        pass",
            },
            harness.ctx,
        )
        assert outcome.is_error
        assert "line" in outcome.content.lower() and "similarity" in outcome.content.lower()

    def test_replace_all_replaces_every_exact_occurrence(
        self, make_ctx, tmp_path: Path
    ) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "old\nkeep\nold\nold\n")

        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "old", "new_string": "new", "replace_all": True},
            harness.ctx,
        )

        assert not outcome.is_error, outcome.content
        assert target.read_text(encoding="utf-8") == "new\nkeep\nnew\nnew\n"

    def test_replace_all_is_refused_for_an_inexact_match(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # Replacing many places on the strength of a fuzzy match is how an agent
        # destroys a file; the model is told to be specific instead.
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "class A:\n\tdef go(self):\n\t\treturn 1\n")
        before = target.read_bytes()

        outcome = EditFileTool().invoke(
            {
                "path": "app.py",
                "old_string": "    def go(self):\n        return 1",
                "new_string": "    def go(self):\n        return 2",
                "replace_all": True,
            },
            harness.ctx,
        )

        assert outcome.is_error and target.read_bytes() == before

    def test_missing_file_points_at_the_right_tools(self, make_ctx) -> None:
        harness = make_ctx()
        outcome = EditFileTool().invoke(
            {"path": "nope.py", "old_string": "a", "new_string": "b"}, harness.ctx
        )
        assert outcome.is_error and "write_file" in outcome.content

    def test_empty_old_string_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        write(tmp_path / "app.py", "content\n")
        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "", "new_string": "x"}, harness.ctx
        )
        assert outcome.is_error

    def test_path_outside_the_workspace_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        outcome = EditFileTool().invoke(
            {"path": "../escape.py", "old_string": "a", "new_string": "b"}, harness.ctx
        )
        assert outcome.is_error
        assert "outside" in outcome.content.lower() or "workspace" in outcome.content.lower()

    def test_crlf_line_endings_survive_byte_for_byte(self, make_ctx, tmp_path: Path) -> None:
        # An edit that silently converts a Windows file to LF shows up as a
        # whole-file diff in the user's version control.
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "line one\nline two\nline three\n", newline="\r\n")

        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "line two", "new_string": "line 2"}, harness.ctx
        )

        assert not outcome.is_error, outcome.content
        data = target.read_bytes()
        assert data == b"line one\r\nline 2\r\nline three\r\n"

    def test_utf8_bom_survives(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "alpha\nbeta\n", encoding="utf-8-sig")

        EditFileTool().invoke(
            {"path": "app.py", "old_string": "beta", "new_string": "gamma"}, harness.ctx
        )

        assert target.read_bytes().startswith(b"\xef\xbb\xbf")
        assert "gamma" in target.read_text(encoding="utf-8-sig")

    def test_non_ascii_content_round_trips(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, '# 说明：计算总和\ntotal = 0\n')

        outcome = EditFileTool().invoke(
            {"path": "app.py", "old_string": "total = 0", "new_string": "total = 1"}, harness.ctx
        )

        assert not outcome.is_error, outcome.content
        assert target.read_text(encoding="utf-8") == '# 说明：计算总和\ntotal = 1\n'

    def test_binary_file_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        outcome = EditFileTool().invoke(
            {"path": "blob.bin", "old_string": "binary", "new_string": "text"}, harness.ctx
        )
        assert outcome.is_error

    def test_approval_request_shows_the_real_diff_without_writing(
        self, make_ctx, tmp_path: Path
    ) -> None:
        # The prompt has to show what would actually happen, or approving it is
        # a guess. Producing it must not touch the file.
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "def add(a, b):\n    return a - b\n")
        before = target.read_bytes()

        request = EditFileTool().approval_request(
            EditFileParams(path="app.py", old_string="return a - b", new_string="return a + b"),
            harness.ctx,
        )

        assert request is not None
        assert request.detail is not None and "+    return a + b" in request.detail
        assert "app.py" in request.summary
        assert request.signature == "edit_file:app.py"
        assert target.read_bytes() == before

    def test_approval_request_never_raises_on_a_bad_path(self, make_ctx) -> None:
        harness = make_ctx()
        request = EditFileTool().approval_request(
            EditFileParams(path="missing.py", old_string="a", new_string="b"), harness.ctx
        )
        assert request is not None  # degraded, but still answerable


class TestMultiEditTool:
    def test_edits_apply_in_sequence(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "a = 1\nb = 2\n")

        outcome = MultiEditTool().invoke(
            {
                "path": "app.py",
                "edits": [
                    {"old_string": "a = 1", "new_string": "a = 10"},
                    {"old_string": "b = 2", "new_string": "b = 20"},
                ],
            },
            harness.ctx,
        )

        assert not outcome.is_error, outcome.content
        assert target.read_text(encoding="utf-8") == "a = 10\nb = 20\n"

    def test_a_later_edit_sees_an_earlier_one(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "value = 1\n")

        outcome = MultiEditTool().invoke(
            {
                "path": "app.py",
                "edits": [
                    {"old_string": "value = 1", "new_string": "value = 2"},
                    {"old_string": "value = 2", "new_string": "value = 3"},
                ],
            },
            harness.ctx,
        )

        assert not outcome.is_error, outcome.content
        assert target.read_text(encoding="utf-8") == "value = 3\n"

    def test_one_failure_abandons_the_whole_batch(self, make_ctx, tmp_path: Path) -> None:
        # A half-applied batch leaves the file in a state neither the model nor
        # the user asked for, which is worse than applying nothing.
        harness = make_ctx()
        target = tmp_path / "app.py"
        write(target, "a = 1\nb = 2\n")
        before = target.read_bytes()

        outcome = MultiEditTool().invoke(
            {
                "path": "app.py",
                "edits": [
                    {"old_string": "a = 1", "new_string": "a = 10"},
                    {"old_string": "does not exist", "new_string": "x"},
                ],
            },
            harness.ctx,
        )

        assert outcome.is_error
        assert "2" in outcome.content  # names which edit failed
        assert target.read_bytes() == before

    def test_empty_edit_list_is_refused(self, make_ctx, tmp_path: Path) -> None:
        harness = make_ctx()
        write(tmp_path / "app.py", "a = 1\n")
        outcome = MultiEditTool().invoke({"path": "app.py", "edits": []}, harness.ctx)
        assert outcome.is_error

    def test_schema_declares_nested_edit_objects(self) -> None:
        schema = MultiEditTool.spec().input_schema
        edits = schema["properties"]["edits"]  # type: ignore[index]
        assert edits["type"] == "array"
        assert set(edits["items"]["required"]) == {"old_string", "new_string"}


class TestReadTextPreserving:
    @pytest.mark.parametrize(
        ("newline", "expected"),
        [("\n", "\n"), ("\r\n", "\r\n")],
    )
    def test_newline_style_is_detected(
        self, tmp_path: Path, newline: str, expected: str
    ) -> None:
        target = tmp_path / "f.txt"
        write(target, "a\nb\nc\n", newline=newline)
        text, style, _ = read_text_preserving(target)
        assert style == expected
        assert "\r" not in text  # normalised for matching

    def test_bom_is_reported_as_the_encoding(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        write(target, "hello\n", encoding="utf-8-sig")
        _, _, encoding = read_text_preserving(target)
        assert encoding == "utf-8-sig"

    def test_gbk_is_decoded_as_a_fallback(self, tmp_path: Path) -> None:
        # A realistic case on a Chinese-locale Windows machine.
        target = tmp_path / "f.txt"
        target.write_bytes("中文内容\n".encode("gbk"))
        text, _, encoding = read_text_preserving(target)
        assert encoding == "gbk" and "中文内容" in text
