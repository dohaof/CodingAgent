"""Reading, writing, and listing files.

Three tools with one shared theme: every answer is bounded, and every bound is
announced. ``read_file`` pages rather than dumping, ``list_dir`` caps its tree,
and both name what they left out so the model can ask for the rest. A failure is
answered the same way — a missing path comes back with the closest names that do
exist, because a model that can see its own typo fixes it in one turn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, ClassVar

from ..errors import ToolError
from ..types import RiskLevel
from .base import ApprovalRequest, BaseTool, ToolContext, ToolOutcome
from .diffs import diff_stats, unified_diff
from .edit import read_text_preserving, write_text_preserving
from .schema import Doc

__all__ = [
    "IGNORED_DIRS",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
]

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".cagent",
        ".tox",
        ".eggs",
    }
)
"""Directory names never walked by the file, search, and listing tools.

Machine-generated trees dwarf hand-written source and would flood any listing
or search with matches no task cares about. Shared with :mod:`.search` so the
tools agree on what the project actually contains.
"""

_BINARY_SNIFF_BYTES = 8192
_MAX_LINE_CHARS = 2000
_SUGGESTION_LIMIT = 5
_SUGGESTION_SCAN_LIMIT = 5000
_SUGGESTION_THRESHOLD = 0.6
"""Minimum name similarity worth suggesting; below this the list is noise."""
_NEW_FILE_PREVIEW_LINES = 40
_LINE_NUMBER_WIDTH = 6


def _looks_binary(path: Path) -> bool:
    """Whether the file's first block contains a NUL byte."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False


def _number_lines(lines: list[str], first_number: int) -> list[str]:
    """Render lines ``cat -n`` style, clipping any pathologically long one."""
    rendered: list[str] = []
    for offset, line in enumerate(lines):
        body = line
        if len(body) > _MAX_LINE_CHARS:
            body = body[:_MAX_LINE_CHARS] + "... [line truncated]"
        rendered.append(f"{first_number + offset:>{_LINE_NUMBER_WIDTH}}\t{body}")
    return rendered


def _similar_paths(target: Path, workspace: Path) -> list[Path]:
    """Workspace files whose name resembles ``target``'s.

    Ranked by name similarity rather than substring containment, because the
    mistakes this catches are typos and half-remembered names — ``aap.py`` for
    ``app.py`` shares no useful substring but is one transposition away.

    The walk is bounded in both breadth and results: a suggestion list is a
    convenience, and a tool that hangs scanning a huge tree to build one has
    made the failure worse than it was.
    """
    wanted = target.name.lower()
    if not wanted:
        return []

    scored: list[tuple[float, Path]] = []
    scanned = 0
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(wanted)
    for root, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            scanned += 1
            if scanned > _SUGGESTION_SCAN_LIMIT:
                scored.sort(key=lambda item: (-item[0], str(item[1])))
                return [path for _, path in scored[:_SUGGESTION_LIMIT]]
            lowered = filename.lower()
            matcher.set_seq1(lowered)
            if matcher.real_quick_ratio() < _SUGGESTION_THRESHOLD:
                continue
            # An identical name in another directory is the likeliest intent, so
            # it outranks every near-miss regardless of path depth.
            score = 2.0 if lowered == wanted else matcher.ratio()
            if score >= _SUGGESTION_THRESHOLD:
                scored.append((score, Path(root, filename)))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored[:_SUGGESTION_LIMIT]]


def _missing_file_error(raw: str, resolved: Path, ctx: ToolContext) -> ToolError:
    """A not-found error carrying the nearest real paths as a hint."""
    suggestions = _similar_paths(resolved, ctx.workspace)
    if suggestions:
        listed = "\n".join(f"  {ctx.rel(path)}" for path in suggestions)
        hint = f"Files with similar names exist:\n{listed}"
    else:
        hint = "Use glob_files or list_dir to find the correct path."
    return ToolError(f"{raw} does not exist.", hint)


@dataclass
class ReadFileParams:
    """Arguments for :class:`ReadFileTool`."""

    path: Annotated[str, Doc("Path to the file, absolute or workspace-relative.")]
    offset: Annotated[int, Doc("1-based line number to start reading from.")] = 1
    limit: Annotated[int, Doc("Maximum number of lines to return.")] = 400


class ReadFileTool(BaseTool):
    """Read a slice of a text file, with line numbers."""

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a text file and return its lines with 1-based line numbers. "
        "Reads a bounded window: pass offset and limit to page through a large "
        "file. The line numbers are what edit_file's old_string should be "
        "copied against."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE
    parallel_safe: ClassVar[bool] = True
    Params: ClassVar[type] = ReadFileParams

    def run(self, params: ReadFileParams, ctx: ToolContext) -> ToolOutcome:
        path = ctx.resolve_path(params.path)
        if path.is_dir():
            raise ToolError(
                f"{ctx.rel(path)} is a directory, not a file.",
                "Use list_dir to see what it contains.",
            )
        if not path.exists():
            raise _missing_file_error(params.path, path, ctx)
        if _looks_binary(path):
            raise ToolError(
                f"{ctx.rel(path)} appears to be a binary file.",
                "Only text files can be read.",
            )

        text, _, _ = read_text_preserving(path)
        if not text:
            return ToolOutcome.ok(
                "(empty file)",
                metadata={"path": ctx.rel(path), "lines_shown": 0, "total_lines": 0},
            )

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)

        start = max(params.offset, 1)
        limit = max(params.limit, 1)
        if start > total:
            raise ToolError(
                f"offset {start} is past the end of {ctx.rel(path)}, which has {total} lines.",
                f"Read from an offset between 1 and {total}.",
            )

        window = lines[start - 1 : start - 1 + limit]
        body = "\n".join(_number_lines(window, start))
        remaining = total - (start - 1 + len(window))
        if remaining > 0:
            next_offset = start + len(window)
            body += (
                f"\n... [{remaining} more lines; "
                f"call read_file with offset={next_offset} to continue]"
            )

        return ToolOutcome.ok(
            body,
            metadata={
                "path": ctx.rel(path),
                "lines_shown": len(window),
                "total_lines": total,
            },
            truncated=remaining > 0,
        )


@dataclass
class WriteFileParams:
    """Arguments for :class:`WriteFileTool`."""

    path: Annotated[str, Doc("Path to write, absolute or workspace-relative.")]
    content: Annotated[str, Doc("The complete new contents of the file.")]


class WriteFileTool(BaseTool):
    """Create a file, or replace one entirely."""

    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Write a complete file, creating parent directories as needed. This "
        "replaces the whole file, so prefer edit_file for changes to an "
        "existing file: a rewrite risks dropping code that was not read."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.MUTATING
    Params: ClassVar[type] = WriteFileParams

    def run(self, params: WriteFileParams, ctx: ToolContext) -> ToolOutcome:
        path = ctx.resolve_path(params.path)
        if path.exists() and not path.is_file():
            raise ToolError(
                f"{ctx.rel(path)} exists and is not a regular file.",
                "Choose a different path.",
            )

        existed = path.is_file()
        before = ""
        newline_style = "\n"
        encoding = "utf-8"
        if existed:
            before, newline_style, encoding = read_text_preserving(path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(
                f"Could not create the parent directory of {ctx.rel(path)}: {exc}"
            ) from exc

        write_text_preserving(path, params.content, newline_style, encoding)

        line_count = len(params.content.splitlines())
        if existed:
            diff = unified_diff(before, params.content, ctx.rel(path))
            added, removed = diff_stats(diff)
            summary = f"Wrote {ctx.rel(path)} ({line_count} lines, +{added}/-{removed})"
            display = diff
        else:
            summary = f"Wrote {ctx.rel(path)} ({line_count} lines, new file)"
            display = self._preview(params.content)

        return ToolOutcome.ok(
            summary,
            display=display,
            metadata={"path": ctx.rel(path), "lines": line_count, "created": not existed},
        )

    @staticmethod
    def _preview(content: str) -> str:
        """The first lines of new content, for the human's benefit."""
        lines = content.split("\n")
        head = lines[:_NEW_FILE_PREVIEW_LINES]
        rendered = "\n".join(_number_lines(head, 1))
        if len(lines) > _NEW_FILE_PREVIEW_LINES:
            rendered += f"\n... [{len(lines) - _NEW_FILE_PREVIEW_LINES} more lines]"
        return rendered

    def approval_request(
        self, params: WriteFileParams, ctx: ToolContext
    ) -> ApprovalRequest | None:
        """Describe the write, showing a real diff when overwriting."""
        try:
            path = ctx.resolve_path(params.path)
        except ToolError:
            # Let run() raise the real error; approving is harmless because the
            # call fails before touching anything.
            return ApprovalRequest(
                tool=self.name,
                risk=self.risk,
                summary=f"write {params.path}",
                signature=f"write_file:{params.path}",
            )

        rel = ctx.rel(path)
        line_count = len(params.content.splitlines())
        if path.is_file():
            try:
                before, _, _ = read_text_preserving(path)
            except ToolError:
                before = ""
            diff = unified_diff(before, params.content, rel)
            added, removed = diff_stats(diff)
            summary = f"overwrite {rel} (+{added}/-{removed})"
            detail = diff
        else:
            summary = f"write {rel} (new file, {line_count} lines)"
            detail = self._preview(params.content)

        return ApprovalRequest(
            tool=self.name,
            risk=self.risk,
            summary=summary,
            detail=detail,
            signature=f"write_file:{rel}",
        )

    def preview(self, params: WriteFileParams, ctx: ToolContext) -> str | None:
        request = self.approval_request(params, ctx)
        return request.detail if request else None


@dataclass
class ListDirParams:
    """Arguments for :class:`ListDirTool`."""

    path: Annotated[str, Doc("Directory to list, absolute or workspace-relative.")] = "."
    depth: Annotated[int, Doc("How many levels to descend.")] = 2
    max_entries: Annotated[int, Doc("Maximum number of entries to return.")] = 200


class ListDirTool(BaseTool):
    """Show a directory tree."""

    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = (
        "List a directory as an indented tree, descending a bounded number of "
        "levels. Generated directories such as .git and node_modules are shown "
        "but not expanded. Use this to orient in an unfamiliar project."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE
    parallel_safe: ClassVar[bool] = True
    Params: ClassVar[type] = ListDirParams

    def run(self, params: ListDirParams, ctx: ToolContext) -> ToolOutcome:
        root = ctx.resolve_path(params.path)
        if not root.exists():
            raise ToolError(
                f"{params.path} does not exist.",
                "Use list_dir on a parent directory to see what is there.",
            )
        if not root.is_dir():
            raise ToolError(
                f"{ctx.rel(root)} is a file, not a directory.",
                "Use read_file to read it.",
            )

        lines: list[str] = [f"{ctx.rel(root)}/"]
        budget = max(params.max_entries, 1)
        shown, omitted = self._walk(root, max(params.depth, 1), 1, budget, lines, ctx)
        if omitted:
            lines.append(f"... [{omitted} more entries omitted]")

        return ToolOutcome.ok(
            "\n".join(lines),
            metadata={"path": ctx.rel(root), "entries": shown},
            truncated=omitted > 0,
        )

    def _walk(
        self,
        directory: Path,
        depth: int,
        level: int,
        budget: int,
        lines: list[str],
        ctx: ToolContext,
    ) -> tuple[int, int]:
        """Append one directory level; return (entries shown, entries omitted)."""
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (p.is_symlink() or not p.is_dir(), p.name.lower()),
            )
        except OSError as exc:
            lines.append(f"{'  ' * level}[unreadable: {exc.strerror or exc}]")
            return 0, 0

        shown = 0
        omitted = 0
        indent = "  " * level
        for entry in entries:
            if shown >= budget:
                omitted += 1
                continue
            try:
                ctx.resolve_path(str(entry))
            except ToolError:
                lines.append(f"{indent}{entry.name}@ [outside workspace]")
                shown += 1
                continue
            # ``is_dir`` follows symlinks. Treat links as leaf entries so a
            # workspace link cannot make list_dir traverse an outside tree.
            is_link = entry.is_symlink()
            is_dir = not is_link and entry.is_dir()
            if is_link:
                lines.append(f"{indent}{entry.name}@")
                shown += 1
                continue
            if is_dir and entry.name in IGNORED_DIRS:
                lines.append(f"{indent}{entry.name}/ …")
                shown += 1
                continue
            lines.append(f"{indent}{entry.name}{'/' if is_dir else ''}")
            shown += 1
            if is_dir and level < depth:
                sub_shown, sub_omitted = self._walk(
                    entry, depth, level + 1, budget - shown, lines, ctx
                )
                shown += sub_shown
                omitted += sub_omitted
        return shown, omitted
