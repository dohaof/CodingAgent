"""Diff rendering and post-edit snippets for the edit tools.

:func:`unified_diff` is what the human approving an edit sees;
:func:`snippet_around` is what the model sees after the edit, so it can verify
the new state without spending a turn re-reading the file. Standard library
only.
"""

from __future__ import annotations

import difflib

__all__ = [
    "diff_stats",
    "snippet_around",
    "unified_diff",
]

_NO_NEWLINE_MARKER = "\\ No newline at end of file"

_LINE_NUMBER_WIDTH = 6


def _diff_lines(text: str) -> list[str]:
    """Split for diffing, making a missing final newline explicit.

    difflib merges an unterminated last line into whatever follows it in the
    output, corrupting the hunk; terminating it and appending a marker line
    keeps the diff well-formed and shows the human that the terminator changed.
    """
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
        lines.append(_NO_NEWLINE_MARKER + "\n")
    return lines


def unified_diff(before: str, after: str, path: str, *, context: int = 3) -> str:
    """A git-style unified diff between two versions of one file.

    Args:
        before: Original text, LF-normalised.
        after: Edited text, LF-normalised.
        path: Workspace-relative path, rendered as ``a/<path>`` / ``b/<path>``.
        context: Unchanged lines shown around each hunk.

    Returns:
        The diff text, or ``""`` when the versions are identical.
    """
    produced = difflib.unified_diff(
        _diff_lines(before),
        _diff_lines(after),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(produced)


def diff_stats(diff_text: str) -> tuple[int, int]:
    """Count changed lines in a unified diff.

    Args:
        diff_text: Output of :func:`unified_diff`.

    Returns:
        ``(added, removed)``. Headers and no-newline markers are not counted.
    """
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")) or line[1:] == _NO_NEWLINE_MARKER:
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def snippet_around(text: str, char_start: int, char_end: int, *, context_lines: int = 4) -> str:
    """A line-numbered excerpt of ``text`` around a char range.

    Returned to the model after an edit so it sees the post-edit state — line
    numbers included, so follow-up edits can be located — without re-reading
    the whole file.

    Args:
        text: The full (post-edit) text.
        char_start: First char of the region of interest.
        char_end: One past its last char; may equal ``char_start``.
        context_lines: Lines shown before and after the region.

    Returns:
        Lines rendered as a 6-wide 1-based line number, a tab, then the line.
    """
    lines = text.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    char_start = max(0, min(char_start, len(text)))
    char_end = max(char_start, min(char_end, len(text)))

    first = text.count("\n", 0, char_start)
    # For a non-empty range, the line holding the last char, not the line a
    # trailing newline would start.
    last_probe = char_end - 1 if char_end > char_start else char_end
    last = text.count("\n", 0, last_probe)
    first = min(first, len(lines) - 1)
    last = min(max(last, first), len(lines) - 1)

    low = max(0, first - context_lines)
    high = min(len(lines), last + context_lines + 1)
    return "\n".join(
        f"{index + 1:>{_LINE_NUMBER_WIDTH}}\t{lines[index]}" for index in range(low, high)
    )
