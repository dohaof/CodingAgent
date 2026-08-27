"""The edit tools: precise local changes instead of whole-file rewrites.

``edit_file`` replaces one block located by :func:`~.matching.find_block`'s
degradation ladder (exact → whitespace → fuzzy); ``multi_edit`` applies a
sequence of such replacements all-or-nothing. Both share a dry-run planner, so
the approval prompt shows the *real* diff of what would be written, and both
write atomically while preserving the file's newline style and encoding — an
edit never converts a CRLF file to LF or strips a BOM as a side effect.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

from ..errors import ToolError, ToolExecutionError
from ..types import RiskLevel
from .base import ApprovalRequest, BaseTool, ToolContext, ToolOutcome
from .diffs import diff_stats, snippet_around, unified_diff
from .matching import (
    Strategy,
    best_rejected,
    describe_ambiguity,
    find_block,
    line_number,
    replace_block,
)
from .schema import Doc

__all__ = [
    "EditFileTool",
    "EditOp",
    "MultiEditTool",
    "read_text_preserving",
    "write_text_preserving",
]

_UTF8_BOM = b"\xef\xbb\xbf"


def read_text_preserving(path: Path) -> tuple[str, str, str]:
    """Read a text file, recording what must survive a round trip.

    Args:
        path: The file to read.

    Returns:
        The LF-normalised text, the dominant newline style (``"\\r\\n"`` or
        ``"\\n"``), and the encoding (``utf-8``, ``utf-8-sig``, or ``gbk``).

    Raises:
        ToolExecutionError: If the file cannot be read, looks binary, or does
            not decode in any supported encoding.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ToolExecutionError(f"Could not read {path}: {exc}") from exc
    if b"\x00" in data:
        raise ToolExecutionError(
            f"{path} appears to be binary (it contains NUL bytes).",
            "Only text files can be edited.",
        )

    encoding = "utf-8-sig" if data.startswith(_UTF8_BOM) else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        try:
            text = data.decode("gbk")
            encoding = "gbk"
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"{path} could not be decoded as UTF-8 or GBK.",
                "This file is not editable as text.",
            ) from exc

    crlf = text.count("\r\n")
    bare_lf = text.count("\n") - crlf
    newline_style = "\r\n" if crlf > bare_lf else "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n"), newline_style, encoding


def write_text_preserving(path: Path, text: str, newline_style: str, encoding: str) -> None:
    """Write LF-normalised text back, restoring style, atomically.

    The bytes go to a temporary file in the same directory and land via
    ``os.replace``, so a crash mid-write leaves the original intact and other
    readers only ever see the old or the new version.

    Args:
        path: Destination file.
        text: LF-normalised content.
        newline_style: As returned by :func:`read_text_preserving`.
        encoding: As returned by :func:`read_text_preserving`;
            ``utf-8-sig`` re-emits the BOM.

    Raises:
        ToolExecutionError: If the temporary file or the replace fails.
    """
    if newline_style != "\n":
        text = text.replace("\n", newline_style)
    payload = text.encode(encoding)
    try:
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
    except OSError as exc:
        raise ToolExecutionError(
            f"Could not create a temporary file next to {path}: {exc}"
        ) from exc
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise ToolExecutionError(f"Could not write {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class _Applied:
    """One replacement that a plan performed, located in the post-edit text."""

    strategy: Strategy
    similarity: float
    start: int
    end: int


@dataclass(slots=True)
class _EditPlan:
    """A fully computed edit that has not touched the disk yet.

    Built identically by :meth:`run` and :meth:`approval_request`, which is
    what lets the approval prompt show the exact diff that a subsequent run
    will write.
    """

    path: Path
    rel: str
    before: str
    after: str
    newline_style: str
    encoding: str
    applied: list[_Applied]

    def diff(self) -> str:
        return unified_diff(self.before, self.after, self.rel)


def _apply_edit(
    text: str,
    old: str,
    new: str,
    *,
    replace_all: bool,
    threshold: float,
    where: str,
) -> tuple[str, list[_Applied]]:
    """Locate ``old`` in ``text`` and splice ``new`` over it, in memory.

    Raises:
        ToolExecutionError: With model-actionable feedback on an empty or
            no-op needle, no match (quoting the closest near-miss when one
            exists), an ambiguous match, or ``replace_all`` over a non-exact
            match.
    """
    if not old:
        raise ToolExecutionError(
            "old_string is empty.",
            "edit_file replaces existing text; use write_file to create or overwrite a file.",
        )
    if old == new:
        raise ToolExecutionError(
            "old_string and new_string are identical, so this edit would change nothing.",
            "Provide a new_string that differs from old_string.",
        )

    matches = find_block(text, old, fuzzy_threshold=threshold)
    if not matches:
        rejected = best_rejected(text, old, fuzzy_threshold=threshold)
        if rejected is not None:
            excerpt = rejected.matched_text.split("\n", 1)[0].strip()
            hint = (
                f"Closest match at line {line_number(text, rejected.start)} "
                f"(similarity {rejected.similarity:.2f}): {excerpt}… — "
                "re-read the file and copy the exact text, including whitespace."
            )
        else:
            hint = "Re-read the file and copy the exact text, including whitespace."
        raise ToolExecutionError(f"old_string was not found in {where}.", hint)

    if len(matches) > 1 and not replace_all:
        raise ToolExecutionError(
            f"old_string is ambiguous in {where}.\n{describe_ambiguity(matches, text)}"
        )

    if replace_all:
        first = matches[0]
        if first.strategy != "exact":
            raise ToolExecutionError(
                f"replace_all only works with exact matches, but old_string matched {where} "
                f"via {first.strategy} matching (similarity {first.similarity:.2f}).",
                "Copy the exact file text into old_string, or apply the edits one at a time.",
            )
        # Splice right-to-left so earlier offsets stay valid while replacing.
        new_text = text
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            new_text = replace_block(new_text, match, new)
        applied: list[_Applied] = []
        shift = 0
        for match in matches:  # exact matches arrive in ascending position order
            start = match.start + shift
            applied.append(_Applied("exact", 1.0, start, start + len(new)))
            shift += len(new) - (match.end - match.start)
        return new_text, applied

    match = matches[0]
    replacement = new
    if match.strategy != "exact" and old.endswith("\n") and new.endswith("\n"):
        # Line-aligned matches exclude the window's trailing newline, so the
        # needle's own trailing newline was never part of the matched range;
        # drop the replacement's to keep the splice symmetric.
        replacement = replacement[:-1]
    new_text = replace_block(text, match, replacement)
    spliced_len = len(new_text) - (len(text) - (match.end - match.start))
    return new_text, [
        _Applied(match.strategy, match.similarity, match.start, match.start + spliced_len)
    ]


def _read_for_edit(params_path: str, ctx: ToolContext) -> tuple[Path, str, str, str, str]:
    """Resolve, sanity-check, and read the target file for a plan."""
    resolved = ctx.resolve_path(params_path)
    rel = ctx.rel(resolved)
    if resolved.is_dir():
        raise ToolExecutionError(
            f"{rel} is a directory, not a file.",
            "Give the path of the file to edit.",
        )
    if not resolved.is_file():
        raise ToolExecutionError(
            f"File {rel} does not exist.",
            "Use write_file to create it, or glob_files to locate the file you meant.",
        )
    before, newline_style, encoding = read_text_preserving(resolved)
    return resolved, rel, before, newline_style, encoding


@dataclass(frozen=True, slots=True)
class EditFileParams:
    """Arguments for :class:`EditFileTool`."""

    path: Annotated[str, Doc("File to edit, relative to the workspace root.")]
    old_string: Annotated[
        str,
        Doc(
            "The text to replace. It must match one unique location; include "
            "several surrounding lines of context to pin it down."
        ),
    ]
    new_string: Annotated[str, Doc("The replacement text.")]
    replace_all: Annotated[
        bool,
        Doc("Replace every exact occurrence instead of requiring a unique match."),
    ] = False


class EditFileTool(BaseTool):
    """Replace one block of text in a file, tolerating stale copies."""

    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace old_string with new_string in a file. Matching degrades "
        "gracefully: exact first, then whitespace-insensitive, then fuzzy — so "
        "minor indentation drift or a small typo in old_string still finds the "
        "right block. old_string must identify a unique location; include "
        "several surrounding lines of context. Set replace_all to change every "
        "exact occurrence at once."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.MUTATING
    Params: ClassVar[type] = EditFileParams

    def _plan(self, params: EditFileParams, ctx: ToolContext) -> _EditPlan:
        """Dry-run the edit: everything except the write."""
        resolved, rel, before, newline_style, encoding = _read_for_edit(params.path, ctx)
        after, applied = _apply_edit(
            before,
            params.old_string,
            params.new_string,
            replace_all=params.replace_all,
            threshold=ctx.config.fuzzy_threshold,
            where=rel,
        )
        return _EditPlan(resolved, rel, before, after, newline_style, encoding, applied)

    def run(self, params: EditFileParams, ctx: ToolContext) -> ToolOutcome:
        plan = self._plan(params, ctx)
        write_text_preserving(plan.path, plan.after, plan.newline_style, plan.encoding)

        diff = plan.diff()
        added, removed = diff_stats(diff)
        first = plan.applied[0]
        occurrences = f", {len(plan.applied)} occurrences" if len(plan.applied) > 1 else ""
        header = (
            f"Edited {plan.rel} via {first.strategy} match "
            f"(similarity {first.similarity:.2f}{occurrences})"
        )
        snippet = snippet_around(plan.after, first.start, first.end)
        return ToolOutcome.ok(
            f"{header}\n{snippet}",
            display=diff,
            metadata={
                "path": plan.rel,
                "strategy": first.strategy,
                "similarity": first.similarity,
                "added": added,
                "removed": removed,
            },
        )

    def approval_request(self, params: EditFileParams, ctx: ToolContext) -> ApprovalRequest | None:
        try:
            plan = self._plan(params, ctx)
            diff = plan.diff()
            added, removed = diff_stats(diff)
            return ApprovalRequest(
                tool=self.name,
                risk=self.risk,
                summary=f"edit {plan.rel} (+{added}/-{removed})",
                detail=diff,
                signature=f"edit_file:{plan.rel}",
            )
        except Exception:  # dry-run failure: run() will surface it as a ToolOutcome
            return ApprovalRequest(
                tool=self.name,
                risk=self.risk,
                summary=f"edit {params.path}",
                detail=None,
                signature=f"edit_file:{params.path}",
            )

    def preview(self, params: EditFileParams, ctx: ToolContext) -> str | None:
        try:
            return self._plan(params, ctx).diff() or None
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class EditOp:
    """One replacement within a :class:`MultiEditTool` batch."""

    old_string: Annotated[
        str,
        Doc("The text to replace; must match one unique location at this point in the sequence."),
    ]
    new_string: Annotated[str, Doc("The replacement text.")]


@dataclass(frozen=True, slots=True)
class MultiEditParams:
    """Arguments for :class:`MultiEditTool`."""

    path: Annotated[str, Doc("File to edit, relative to the workspace root.")]
    edits: Annotated[
        list[EditOp],
        Doc("Replacements applied in order; each edit sees the previous edit's output."),
    ]


class MultiEditTool(BaseTool):
    """Apply several replacements to one file as a single atomic change."""

    name: ClassVar[str] = "multi_edit"
    description: ClassVar[str] = (
        "Apply several {old_string, new_string} replacements to one file in "
        "order, each edit seeing the previous edit's output. The whole batch "
        "is atomic: if any edit fails, the file is left untouched. Prefer this "
        "over repeated edit_file calls when making related changes to one file."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.MUTATING
    Params: ClassVar[type] = MultiEditParams

    def _plan(self, params: MultiEditParams, ctx: ToolContext) -> _EditPlan:
        """Dry-run every edit in memory; any failure leaves no trace on disk."""
        if not params.edits:
            raise ToolExecutionError(
                "No edits were provided.",
                "Pass at least one {old_string, new_string} pair in edits.",
            )
        resolved, rel, before, newline_style, encoding = _read_for_edit(params.path, ctx)
        text = before
        applied: list[_Applied] = []
        total = len(params.edits)
        for index, op in enumerate(params.edits):
            try:
                text, step = _apply_edit(
                    text,
                    op.old_string,
                    op.new_string,
                    replace_all=False,
                    threshold=ctx.config.fuzzy_threshold,
                    where=rel,
                )
            except ToolError as exc:
                raise ToolExecutionError(
                    f"Edit {index + 1} of {total} failed; {rel} was left unchanged.\n"
                    f"{exc.message}",
                    exc.hint,
                ) from exc
            applied.extend(step)
        return _EditPlan(resolved, rel, before, text, newline_style, encoding, applied)

    def run(self, params: MultiEditParams, ctx: ToolContext) -> ToolOutcome:
        plan = self._plan(params, ctx)
        write_text_preserving(plan.path, plan.after, plan.newline_style, plan.encoding)

        diff = plan.diff()
        added, removed = diff_stats(diff)
        lines = [f"Applied {len(plan.applied)} edits to {plan.rel}:"]
        lines.extend(
            f"  {index}. {step.strategy} match (similarity {step.similarity:.2f})"
            for index, step in enumerate(plan.applied, start=1)
        )
        return ToolOutcome.ok(
            "\n".join(lines),
            display=diff,
            metadata={
                "path": plan.rel,
                "edits": len(plan.applied),
                "strategies": [step.strategy for step in plan.applied],
                "added": added,
                "removed": removed,
            },
        )

    def approval_request(self, params: MultiEditParams, ctx: ToolContext) -> ApprovalRequest | None:
        try:
            plan = self._plan(params, ctx)
            diff = plan.diff()
            added, removed = diff_stats(diff)
            return ApprovalRequest(
                tool=self.name,
                risk=self.risk,
                summary=f"edit {plan.rel} ({len(plan.applied)} edits, +{added}/-{removed})",
                detail=diff,
                signature=f"multi_edit:{plan.rel}",
            )
        except Exception:  # dry-run failure: run() will surface it as a ToolOutcome
            return ApprovalRequest(
                tool=self.name,
                risk=self.risk,
                summary=f"edit {params.path} ({len(params.edits)} edits)",
                detail=None,
                signature=f"multi_edit:{params.path}",
            )

    def preview(self, params: MultiEditParams, ctx: ToolContext) -> str | None:
        try:
            return self._plan(params, ctx).diff() or None
        except Exception:
            return None
