"""Output budgeting: keep the informative ends, drop the middle, say so.

A build log or a test run can produce thousands of lines, and pasting all of it
into the transcript costs tokens the task still needs later. The head carries
the invocation and the first failure; the tail carries the summary and the exit
status; the middle is usually repetition. So both ends survive and the gap is
replaced by a marker that states exactly how much went missing — a model that
knows output was elided can ask for more, whereas silent truncation reads as a
complete answer and quietly misleads it.

Used by the shell tool for command output and by the context manager when
compacting history, so the two agree on what an elision looks like.
"""

from __future__ import annotations

__all__ = ["truncate_output"]

_LINE_MARKER = "\n... [{omitted} lines omitted] ...\n"
_CHAR_MARKER = "\n... [{omitted} characters omitted] ...\n"


def truncate_output(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_chars: int,
) -> tuple[str, bool]:
    """Trim ``text`` to a line and character budget.

    Args:
        text: The raw output.
        head_lines: Lines to keep from the start.
        tail_lines: Lines to keep from the end.
        max_chars: Ceiling on the returned length, applied after the line trim
            so that a few very long lines cannot defeat the line budget.

    Returns:
        The possibly-trimmed text and whether anything was dropped.
    """
    if not text:
        return text, False

    result = text
    truncated = False

    lines = text.split("\n")
    keep = max(head_lines, 0) + max(tail_lines, 0)
    if keep and len(lines) > keep:
        omitted = len(lines) - keep
        head = lines[: max(head_lines, 0)]
        tail = lines[len(lines) - max(tail_lines, 0) :] if tail_lines > 0 else []
        result = "\n".join(head) + _LINE_MARKER.format(omitted=omitted) + "\n".join(tail)
        truncated = True

    if max_chars > 0 and len(result) > max_chars:
        result = _truncate_chars(result, max_chars)
        truncated = True

    return result, truncated


def _truncate_chars(text: str, max_chars: int) -> str:
    """Keep the first and last characters of ``text`` within ``max_chars``.

    Splitting the budget in half rather than keeping only a prefix matters for
    the pathological case this guards: a single multi-megabyte line, where the
    tail holds the result and a prefix-only cut would discard it.
    """
    marker_width = len(_CHAR_MARKER.format(omitted=len(text)))
    room = max_chars - marker_width
    if room <= 0:
        return text[:max_chars]

    head_chars = room // 2
    tail_chars = room - head_chars
    omitted = len(text) - head_chars - tail_chars
    head = text[:head_chars]
    tail = text[len(text) - tail_chars :] if tail_chars else ""
    return head + _CHAR_MARKER.format(omitted=omitted) + tail
