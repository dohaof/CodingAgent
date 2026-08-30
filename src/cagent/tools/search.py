"""Finding things: file names by glob, file contents by regex.

These are what let the agent work like a programmer in an unfamiliar repository
— search first, read second — instead of asking for whole files and hoping the
relevant lines are in the window. Everything is bounded and every bound is
announced, and finding nothing is a successful result with a plain answer, not
an error: "no matches" is information the model needs to trust.

``grep_search`` prefers ripgrep when it is installed and falls back to a
hand-written walker when it is not. Both engines emit byte-identical output, so
a task behaves the same on a machine without ripgrep — only slower.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

from ..errors import ToolError
from ..types import RiskLevel
from .base import BaseTool, ToolContext, ToolOutcome
from .files import IGNORED_DIRS
from .schema import Doc

__all__ = ["GlobFilesTool", "GrepSearchTool"]

_GLOB_RESULT_CAP = 100
_LINE_CLIP_CHARS = 300
_RIPGREP_TIMEOUT = 30.0
_BINARY_SNIFF_BYTES = 8192
_MAX_FILE_BYTES = 4_000_000
"""Files larger than this are skipped by the fallback engine: a multi-megabyte
generated blob costs seconds to scan and never holds the answer."""

_MATCH_SEP = "\x1f"
_CONTEXT_SEP = "\x0e"
"""Field separators requested of ripgrep: control characters that cannot occur
in source text, which makes its output unambiguous to split.

Neither is one of the characters :meth:`str.splitlines` treats as a line
boundary — that rules out the otherwise obvious record separator ``\\x1e``,
which would be swallowed as a newline before the parse ever saw it. The output
is split on ``\\n`` explicitly for the same reason."""

_RESULT_LINE_RE = re.compile(r"^(.+):(\d+)([:-])\s(.*)$")


def _parse_ripgrep_line(raw: str) -> tuple[str, str, str, bool] | None:
    """Split one ripgrep output line into path, line number, text, and kind.

    Returns ``None`` for a line in neither field format, which is how a
    diagnostic or an unrecognised form gets skipped instead of corrupting the
    results.
    """
    for separator, is_match in ((_MATCH_SEP, True), (_CONTEXT_SEP, False)):
        if separator not in raw:
            continue
        path, _, rest = raw.partition(separator)
        number, found, text = rest.partition(separator)
        if found and number.isdigit():
            return path, number, text, is_match
    return None


def _human_size(byte_count: int) -> str:
    """Render a file size compactly."""
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _is_ignored(path: Path, base: Path) -> bool:
    """Whether any component below ``base`` is a generated directory."""
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        parts = path.parts
    return any(part in IGNORED_DIRS for part in parts)


def _has_symlink_component(path: Path, root: Path) -> bool:
    """Return whether a candidate reaches a file through a symlink.

    ``Path.is_file`` and ``open`` follow links. Skipping every linked component
    keeps discovery tools from reading a file outside the workspace through a
    link, including links to directories that ``Path.glob`` may traverse.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _clip(line: str) -> str:
    """Bound one output line."""
    text = line.rstrip("\n\r")
    if len(text) > _LINE_CLIP_CHARS:
        return text[:_LINE_CLIP_CHARS] + "…"
    return text


def _group_search_lines(lines: list[str]) -> list[str]:
    """Show each matched file once, followed by its numbered excerpts.

    Keeping the path on a heading makes a long search result substantially
    easier to scan while preserving ``:`` for matching lines and ``-`` for
    surrounding context.  Non-result lines (for example the capped notice)
    stay where the search engine emitted them.
    """
    grouped: list[str] = []
    current_path: str | None = None
    for line in lines:
        match = _RESULT_LINE_RE.match(line)
        if match is None:
            grouped.append(line)
            continue
        path, number, separator, text = match.groups()
        if path != current_path:
            if grouped and grouped[-1] != "":
                grouped.append("")
            grouped.append(path)
            current_path = path
        grouped.append(f"    {number}{separator} {text}")
    return grouped


@dataclass
class GlobFilesParams:
    """Arguments for :class:`GlobFilesTool`."""

    pattern: Annotated[str, Doc("Glob pattern, for example **/*.py or src/**/test_*.py.")]
    path: Annotated[str, Doc("Directory to search from.")] = "."


class GlobFilesTool(BaseTool):
    """Find files by name pattern."""

    name: ClassVar[str] = "glob_files"
    description: ClassVar[str] = (
        "Find files whose path matches a glob pattern, most recently modified "
        "first. Use it to locate a file when you know roughly what it is called. "
        "Generated directories such as .git and node_modules are skipped."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE
    parallel_safe: ClassVar[bool] = True
    Params: ClassVar[type] = GlobFilesParams

    def run(self, params: GlobFilesParams, ctx: ToolContext) -> ToolOutcome:
        base = ctx.resolve_path(params.path)
        if not base.is_dir():
            raise ToolError(
                f"{params.path} is not a directory.",
                "Give a directory to search from, such as '.'.",
            )

        pattern = params.pattern.strip()
        if not pattern:
            raise ToolError("No glob pattern was given.", "For example: **/*.py")

        try:
            candidates = list(base.glob(pattern))
        except (ValueError, OSError, NotImplementedError) as exc:
            raise ToolError(
                f"{pattern!r} is not a usable glob pattern: {exc}",
                "Use shell-style globs such as **/*.py or src/*.ts.",
            ) from exc

        files: list[Path] = []
        for candidate in candidates:
            try:
                ctx.resolve_path(str(candidate))
            except ToolError:
                continue
            if (
                not _has_symlink_component(candidate, base)
                and candidate.is_file()
                and not _is_ignored(candidate, base)
            ):
                files.append(candidate)
        if not files:
            return ToolOutcome.ok(
                f"No files match {pattern!r} under {ctx.rel(base)}.",
                metadata={"pattern": pattern, "matches": 0},
            )

        files.sort(key=lambda p: (-self._mtime(p), str(p)))
        shown = files[:_GLOB_RESULT_CAP]
        lines = [f"{ctx.rel(path)}  ({_human_size(self._size(path))})" for path in shown]
        if len(files) > _GLOB_RESULT_CAP:
            lines.append(f"... and {len(files) - _GLOB_RESULT_CAP} more")

        return ToolOutcome.ok(
            "\n".join(lines),
            metadata={"pattern": pattern, "matches": len(files)},
            truncated=len(files) > _GLOB_RESULT_CAP,
        )

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0


@dataclass
class GrepSearchParams:
    """Arguments for :class:`GrepSearchTool`."""

    pattern: Annotated[str, Doc("Regular expression to search for.")]
    path: Annotated[str, Doc("File or directory to search.")] = "."
    glob: Annotated[
        str | None, Doc("Restrict the search to files matching this glob, e.g. *.py.")
    ] = None
    case_sensitive: Annotated[bool, Doc("Match case exactly.")] = False
    context: Annotated[int, Doc("Lines of surrounding context to include.")] = 0
    max_results: Annotated[int, Doc("Maximum number of matching lines to return.")] = 60


class GrepSearchTool(BaseTool):
    """Search file contents by regular expression."""

    name: ClassVar[str] = "grep_search"
    description: ClassVar[str] = (
        "Search file contents with a regular expression and return matching "
        "lines grouped by file with line numbers and text. Use it to find a "
        "symbol, a call site, or a string before reading any file. "
        "Results are capped, so narrow the "
        "pattern or pass glob to focus the search."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE
    parallel_safe: ClassVar[bool] = True
    Params: ClassVar[type] = GrepSearchParams

    def run(self, params: GrepSearchParams, ctx: ToolContext) -> ToolOutcome:
        pattern = params.pattern
        if not pattern:
            raise ToolError("No search pattern was given.", "Pass a regular expression.")

        base = ctx.resolve_path(params.path)
        if not base.exists():
            raise ToolError(
                f"{params.path} does not exist.",
                "Use list_dir or glob_files to find the right path.",
            )

        # Compiled up front even when ripgrep will run, so an invalid pattern
        # fails with Python's specific message instead of ripgrep's exit code.
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ToolError(
                f"{pattern!r} is not a valid regular expression: {exc}",
                "Escape regex metacharacters such as ( ) [ ] * + ? to match them literally.",
            ) from exc

        limit = max(params.max_results, 1)
        context = max(params.context, 0)
        engine = "ripgrep" if shutil.which("rg") else "python"

        if engine == "ripgrep":
            hits = self._ripgrep(params, base, limit, context)
            if hits is None:
                engine = "python"
        if engine == "python":
            hits = self._python_search(params, base, limit, context, ctx)

        assert hits is not None  # both branches assign, or fall through to python
        if not hits.lines:
            return ToolOutcome.ok(
                f"No matches for {pattern!r} in {ctx.rel(base)}.",
                metadata={"engine": engine, "matches": 0},
            )

        body = "\n".join(_group_search_lines(hits.lines))
        if hits.capped:
            body += f"\n[capped at {limit} results — refine the pattern or add a glob filter]"

        return ToolOutcome.ok(
            body,
            metadata={"engine": engine, "matches": hits.matches},
            truncated=hits.capped,
        )

    def _ripgrep(
        self, params: GrepSearchParams, base: Path, limit: int, context: int
    ) -> _Hits | None:
        """Search with ripgrep, or return ``None`` to fall back.

        Ripgrep runs with the search root as its working directory, so paths
        come back relative and a Windows drive letter never reaches the parse.

        The field separators are overridden to control characters rather than
        left at ``:`` and ``-``. With the defaults, a context line whose text
        contains something like ``:12:`` is indistinguishable from a match line,
        and the parse has to guess; control characters cannot occur in source
        text, so splitting becomes exact. An older ripgrep that rejects the
        flags exits non-zero, which falls back to the Python engine.
        """
        argv = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--no-config",
            "--no-follow",
            f"--field-match-separator={_MATCH_SEP}",
            f"--field-context-separator={_CONTEXT_SEP}",
        ]
        if not params.case_sensitive:
            argv.append("-i")
        if context:
            argv += ["-C", str(context)]
        if params.glob:
            argv += ["--glob", params.glob]
        for ignored in sorted(IGNORED_DIRS):
            argv += ["--glob", f"!{ignored}/**"]

        searching_file = base.is_file()
        cwd = base.parent if searching_file else base
        argv += ["--", params.pattern, base.name if searching_file else "."]

        try:
            completed = subprocess.run(  # noqa: S603  # fixed argv, no shell
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_RIPGREP_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode not in (0, 1):
            return None

        lines: list[str] = []
        matches = 0
        capped = False
        for line in completed.stdout.split("\n"):
            raw = line.rstrip("\r")
            if raw == "--":
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            parsed = _parse_ripgrep_line(raw)
            if parsed is None:
                continue
            path, number, text, is_match = parsed
            if is_match:
                if matches >= limit:
                    capped = True
                    break
                matches += 1
            separator = ":" if is_match else "-"
            shown = base.name if searching_file else path.replace("\\", "/").removeprefix("./")
            lines.append(f"{shown}:{number}{separator} {_clip(text)}")

        while lines and lines[-1] == "":
            lines.pop()
        return _Hits(lines, matches, capped)

    def _python_search(
        self,
        params: GrepSearchParams,
        base: Path,
        limit: int,
        context: int,
        ctx: ToolContext,
    ) -> _Hits:
        """Search with the standard library, for machines without ripgrep."""
        flags = 0 if params.case_sensitive else re.IGNORECASE
        regex = re.compile(params.pattern, flags)

        lines: list[str] = []
        matches = 0
        capped = False
        first_file = True
        for path in self._candidate_files(params, base):
            try:
                ctx.resolve_path(str(path))
            except ToolError:
                continue
            text = self._read_text(path)
            if text is None:
                continue
            file_lines = text.split("\n")
            found = [i for i, line in enumerate(file_lines) if regex.search(line)]
            if not found:
                continue

            rendered = base.name if base.is_file() else Path(path).relative_to(base).as_posix()
            if not first_file:
                lines.append("")
            first_file = False

            emitted: set[int] = set()
            for index in found:
                if matches >= limit:
                    capped = True
                    break
                matches += 1
                low = max(index - context, 0)
                high = min(index + context, len(file_lines) - 1)
                for position in range(low, high + 1):
                    if position in emitted:
                        continue
                    emitted.add(position)
                    separator = ":" if position == index else "-"
                    lines.append(
                        f"{rendered}:{position + 1}{separator} {_clip(file_lines[position])}"
                    )
            if capped:
                break

        while lines and lines[-1] == "":
            lines.pop()
        return _Hits(lines, matches, capped)

    @staticmethod
    def _candidate_files(params: GrepSearchParams, base: Path) -> list[Path]:
        """Files to scan, in a stable order."""
        if base.is_file():
            return [base]

        selected: list[Path] = []
        for root, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in IGNORED_DIRS and not (Path(root) / d).is_symlink()
            )
            for filename in sorted(filenames):
                path = Path(root, filename)
                if path.is_symlink():
                    continue
                if params.glob and not path.match(params.glob):
                    continue
                selected.append(path)
        return selected

    @staticmethod
    def _read_text(path: Path) -> str | None:
        """Decode a file, or ``None`` if it is binary, huge, or unreadable."""
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                return None
            with path.open("rb") as handle:
                head = handle.read(_BINARY_SNIFF_BYTES)
                if b"\x00" in head:
                    return None
                rest = handle.read()
        except OSError:
            return None
        return (head + rest).decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class _Hits:
    """One engine's results, before the cap notice is appended."""

    lines: list[str]
    matches: int
    capped: bool
