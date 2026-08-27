"""Block matching for the edit engine: a three-level degradation ladder.

:func:`find_block` locates the text a model asked to replace. It tries three
strategies in order and returns every match from the *first* level that yields
any, never mixing levels:

1. **exact** — plain substring search, the fast common case.
2. **whitespace** — line-by-line comparison after ``str.strip``, which absorbs
   uniform indentation drift, tabs-vs-spaces, and trailing whitespace: the
   classic reasons a verbatim ``old_string`` copied from a stale read no longer
   matches.
3. **fuzzy** — windowed :class:`difflib.SequenceMatcher` scoring over
   trailing-whitespace-stripped text, which tolerates small typos in the
   needle at the cost of an explicit similarity threshold.

Offset convention: exact matches span exactly the needle. Whitespace and fuzzy
matches are line-aligned windows whose ``[start, end)`` range begins at the
first character of the window's first line and ends after the last character
of its last line, *excluding* that line's trailing newline. The newline stays
in the surrounding text, so :func:`replace_block` can splice a replacement
that carries no trailing newline without ever doubling or losing one.

Pure logic over strings — no I/O, no config, standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

__all__ = [
    "MatchResult",
    "Strategy",
    "best_rejected",
    "describe_ambiguity",
    "find_block",
    "line_number",
    "replace_block",
]

Strategy = Literal["exact", "whitespace", "fuzzy"]
"""Which ladder level produced a match."""

_WHITESPACE_SIMILARITY = 0.99
"""Nominal score for whitespace-level matches: below exact, above any fuzzy."""

_AMBIGUITY_BAND = 0.005
"""Fuzzy matches within this of the best are returned too, so a near-tie
surfaces as an ambiguity instead of a silently arbitrary choice."""

_REJECT_FLOOR = 0.4
"""Fuzzy candidates below this are too dissimilar to be worth reporting even
as a rejected near-miss."""

_EXCERPT_CHARS = 80
"""Longest one-line excerpt quoted in ambiguity and rejection messages."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One located occurrence of the needle inside the haystack."""

    start: int
    """Char offset of the first matched character."""

    end: int
    """Char offset one past the last matched character (see module docstring
    for the trailing-newline convention on line-aligned matches)."""

    strategy: Strategy
    similarity: float
    matched_text: str
    """The actual haystack text in ``[start, end)``, which for whitespace and
    fuzzy matches differs from the needle."""

    indent_delta: str | None = None
    """Uniform indentation prefix present in the match but not the needle;
    ``None`` when the indents agree or differ non-uniformly.
    :func:`replace_block` prepends it to every replacement line."""


def _split_lines(text: str) -> tuple[list[str], list[int]]:
    """Split into terminator-free lines plus each line's start offset.

    A trailing newline produces no phantom empty final line, so windows can
    never extend past the real content.
    """
    lines = text.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    return lines, offsets


def _window_span(lines: list[str], offsets: list[int], first: int, count: int) -> tuple[int, int]:
    """Char range of ``lines[first:first + count]``, trailing newline excluded."""
    last = first + count - 1
    return offsets[first], offsets[last] + len(lines[last])


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _first_nonempty(lines: list[str]) -> str | None:
    return next((line for line in lines if line.strip()), None)


def _indent_delta(needle_lines: list[str], window_lines: list[str]) -> str | None:
    """The uniform prefix the match adds to the needle's indentation.

    Compared on the first non-empty line of each side. Only an *added* prefix
    is representable; when the match is less indented than the needle, or the
    two indents share no suffix (e.g. tabs vs spaces), the delta is ``None``
    and the replacement keeps its own indentation.
    """
    needle_line = _first_nonempty(needle_lines)
    window_line = _first_nonempty(window_lines)
    if needle_line is None or window_line is None:
        return None
    needle_indent = _leading_ws(needle_line)
    window_indent = _leading_ws(window_line)
    if window_indent == needle_indent or not window_indent.endswith(needle_indent):
        return None
    delta = window_indent[: len(window_indent) - len(needle_indent)]
    return delta or None


def _exact_matches(haystack: str, needle: str) -> list[MatchResult]:
    """Every non-overlapping literal occurrence, in position order."""
    results: list[MatchResult] = []
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        results.append(MatchResult(start, end, "exact", 1.0, needle))
        start = haystack.find(needle, end)
    return results


def _whitespace_matches(haystack: str, needle: str) -> list[MatchResult]:
    """Non-overlapping line windows equal to the needle after per-line strip."""
    needle_lines, _ = _split_lines(needle)
    stripped_needle = [line.strip() for line in needle_lines]
    hay_lines, offsets = _split_lines(haystack)
    count = len(needle_lines)
    results: list[MatchResult] = []
    index = 0
    while index + count <= len(hay_lines):
        window = hay_lines[index : index + count]
        if all(window[k].strip() == stripped_needle[k] for k in range(count)):
            start, end = _window_span(hay_lines, offsets, index, count)
            results.append(
                MatchResult(
                    start,
                    end,
                    "whitespace",
                    _WHITESPACE_SIMILARITY,
                    haystack[start:end],
                    _indent_delta(needle_lines, window),
                )
            )
            index += count
        else:
            index += 1
    return results


def _fuzzy_candidates(haystack: str, needle: str, floor: float) -> list[MatchResult]:
    """All line windows scoring at least ``floor`` against the needle.

    Windows one line shorter and longer than the needle are tried too, so an
    inserted or deleted line in the needle still finds its block. Scores are
    computed on trailing-whitespace-stripped text; ``real_quick_ratio`` and
    ``quick_ratio`` upper bounds gate the expensive ``ratio`` call.
    """
    hay_lines, offsets = _split_lines(haystack)
    needle_lines, _ = _split_lines(needle)
    if not hay_lines or not needle_lines:
        return []
    stripped = [line.rstrip() for line in hay_lines]
    needle_norm = "\n".join(line.rstrip() for line in needle_lines)
    base = len(needle_lines)
    sizes = sorted({size for size in (base - 1, base, base + 1) if 1 <= size <= len(hay_lines)})

    matcher: SequenceMatcher[str] = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle_norm)
    candidates: list[MatchResult] = []
    for size in sizes:
        for index in range(len(hay_lines) - size + 1):
            matcher.set_seq1("\n".join(stripped[index : index + size]))
            if matcher.real_quick_ratio() < floor or matcher.quick_ratio() < floor:
                continue
            score = matcher.ratio()
            if score < floor:
                continue
            start, end = _window_span(hay_lines, offsets, index, size)
            candidates.append(
                MatchResult(
                    start,
                    end,
                    "fuzzy",
                    score,
                    haystack[start:end],
                    _indent_delta(needle_lines, hay_lines[index : index + size]),
                )
            )
    return candidates


def _fuzzy_matches(haystack: str, needle: str, threshold: float) -> list[MatchResult]:
    """Best fuzzy window plus any near-tie, overlaps suppressed."""
    candidates = _fuzzy_candidates(haystack, needle, threshold)
    if not candidates:
        return []
    candidates.sort(key=lambda m: (-m.similarity, m.start))
    kept: list[MatchResult] = []
    for candidate in candidates:
        if all(candidate.end <= other.start or candidate.start >= other.end for other in kept):
            kept.append(candidate)
    best = kept[0].similarity
    return [match for match in kept if match.similarity >= best - _AMBIGUITY_BAND]


def find_block(haystack: str, needle: str, *, fuzzy_threshold: float) -> list[MatchResult]:
    """Locate ``needle`` in ``haystack`` via the degradation ladder.

    Args:
        haystack: The full text to search, LF-normalised.
        needle: The block to find, LF-normalised.
        fuzzy_threshold: Minimum :class:`difflib.SequenceMatcher` ratio for a
            fuzzy match to count, in ``(0, 1)``.

    Returns:
        Every match from the first ladder level that produced any — levels are
        never mixed. Exact and whitespace matches come in position order;
        fuzzy matches come best-first, and more than one means the best score
        was tied within :data:`_AMBIGUITY_BAND`. Empty when nothing matched.
    """
    if not needle:
        return []
    for level in (_exact_matches, _whitespace_matches):
        found = level(haystack, needle)
        if found:
            return found
    return _fuzzy_matches(haystack, needle, fuzzy_threshold)


def best_rejected(haystack: str, needle: str, *, fuzzy_threshold: float) -> MatchResult | None:
    """The best fuzzy candidate that fell *below* the threshold.

    Used to build actionable no-match errors: the model learns where the
    nearest look-alike lives and how far off its needle was, instead of a bare
    "not found".
    """
    if not needle:
        return None
    floor = min(_REJECT_FLOOR, fuzzy_threshold)
    candidates = _fuzzy_candidates(haystack, needle, floor)
    below = [match for match in candidates if match.similarity < fuzzy_threshold]
    if not below:
        return None
    return max(below, key=lambda m: (m.similarity, -m.start))


def replace_block(haystack: str, match: MatchResult, replacement: str) -> str:
    """Splice ``replacement`` over ``match`` in ``haystack``.

    When ``match.indent_delta`` is set, every non-blank replacement line is
    re-indented by that prefix first. This is what makes a whitespace or fuzzy
    replace *usable*: the needle (and hence the replacement the model wrote)
    carries the indentation of its stale copy, and without the shift the new
    block would land mis-indented relative to the code around it.
    """
    body = replacement
    if match.indent_delta:
        body = "\n".join(
            match.indent_delta + line if line.strip() else line
            for line in replacement.split("\n")
        )
    return haystack[: match.start] + body + haystack[match.end :]


def _excerpt(text: str) -> str:
    """First line of a match, stripped and bounded, for one-line quoting."""
    line = text.split("\n", 1)[0].strip()
    if len(line) > _EXCERPT_CHARS:
        return line[:_EXCERPT_CHARS] + "…"
    return line


def line_number(haystack: str, offset: int) -> int:
    """1-based line number of a char offset."""
    return haystack.count("\n", 0, offset) + 1


def describe_ambiguity(matches: list[MatchResult], haystack: str) -> str:
    """Model-facing description of multiple candidate locations.

    Lists each candidate with its line number and a one-line excerpt, then
    tells the model how to disambiguate — this text goes back into the
    transcript verbatim as a tool result.
    """
    lines = [f"old_string matches {len(matches)} locations:"]
    for match in sorted(matches, key=lambda m: m.start):
        lines.append(
            f"  - line {line_number(haystack, match.start)} "
            f"({match.strategy}, similarity {match.similarity:.2f}): "
            f"{_excerpt(match.matched_text)}"
        )
    lines.append(
        "Add more surrounding lines to old_string so it identifies exactly one "
        "location, or set replace_all=true to replace every exact occurrence."
    )
    return "\n".join(lines)
