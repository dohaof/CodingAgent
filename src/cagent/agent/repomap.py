"""Build a small, task-aware index of a source tree.

The map is deliberately an index rather than a code dump. It gives the model
file paths, declarations, and a small amount of module structure so the model
can choose the right ``read_file`` or ``grep_search`` call. Python uses its
standard AST; other languages use Tree-sitter when a grammar is already
available locally, and fall back to conservative declaration matching.

Tree-sitter is optional. ``tree-sitter-language-pack`` downloads grammars on
first use, which is an unacceptable side effect during Agent startup, so this
module only calls it for languages reported as already downloaded. Users who
install the optional extra and prefetch grammars get syntax-aware parsing;
everyone else still gets a useful path and declaration index.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ..llm.tokens import estimate_text
from ..tools.files import IGNORED_DIRS
from .terms import (
    cjk_bigrams,
    expand_cjk,
    identifier_groups,
    is_stopword,
    prose_text,
    split_identifier,
)

__all__ = ["FileOutline", "RepoMap", "RepoMapIndex", "build_repo_map"]

Layers = Literal["both", "paths", "details"]
"""Which half of the map to render.

``paths`` and ``details`` exist because the two halves have different
lifetimes. The path index depends only on the tree, so it belongs in the
cached system prompt; the detailed entries depend on the task, so they ship
with the task and leave the cached prefix alone.
"""

_MAX_FILES_SCANNED = 2000
_MAX_FILE_BYTES = 400_000
_MAX_SYMBOLS_PER_FILE = 24
_MAX_TEXT_TERMS = 96

_PATH_WEIGHT = 12.0
_SYMBOL_WEIGHT = 9.0
_TEXT_WEIGHT = 2.5
"""What a term match is worth, by where it matched. A term scores once, at its
strongest field, so a name repeated in path and symbols is not counted twice."""

_IDF_FLOOR = 0.5
_IDF_CEILING = 1.6
"""How far inverse document frequency may scale a match. A term in every file
of the project says nothing about which file to read; a term in one file says
almost everything. Bounded on both sides because a corpus of three files has no
meaningful statistics and should not have its ranking dominated by them."""

_PREFIX_WEIGHT = 0.6
_MIN_PREFIX_LENGTH = 4
_MAX_PREFIX_EXPANSIONS = 6
"""Morphology, cheaply. ``pagination`` never equals ``paginate`` and
``cancellation`` never equals ``cancel``, but a shared prefix catches both
without a stemmer, at a discount because a shared prefix is weaker evidence."""

_EXPANSION_WEIGHT = 0.9
"""What a Chinese term's English translation is worth. Nearly full weight: in a
project whose identifiers are English, the translation *is* the usable term."""

_ScoredTerm = tuple[str, float, float, float]
"""One query term with its already-resolved worth in the path, symbol, and text
fields. Resolved once per query so scoring a file is three set lookups."""


@lru_cache(maxsize=16384)
def _cost(text: str, model: str) -> int:
    """Token estimate for one rendered block, memoised.

    Selection prices every candidate two or three times and the map is
    re-rendered whenever the tree changes, so a large project used to spend
    thousands of tokeniser passes per rebuild on strings that had not changed.
    """
    return estimate_text(text, model=model)

_ENTRY_POINT_NAMES = frozenset(
    {
        "main.py",
        "__main__.py",
        "app.py",
        "cli.py",
        "server.py",
        "index.js",
        "index.ts",
        "main.go",
        "main.rs",
        "lib.rs",
        "mod.rs",
        "Main.java",
        "Program.cs",
        "main.swift",
    }
)

_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".astro": "astro",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".dart": "dart",
    ".scala": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
}

_TREE_SITTER_NAMES = {"csharp": "c_sharp"}

# Fallback only. A false negative costs one search; a false positive can make
# the model trust a symbol that does not exist.
_REGEX_DECLARATIONS: dict[str, tuple[re.Pattern[str], ...]] = {
    "python": (re.compile(r"^\s*(?:async\s+)?def\s+\w+"), re.compile(r"^\s*class\s+\w+")),
    "javascript": (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b"),
        re.compile(r"^\s*(?:export\s+)?class\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function\b"),
    ),
    "typescript": (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b"),
        re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"),
    ),
    "tsx": (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b"),
        re.compile(r"^\s*(?:export\s+)?class\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+\w+"),
    ),
    "go": (
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?\w+"),
        re.compile(r"^\s*type\s+\w+\s+(?:struct|interface|func)"),
    ),
    "rust": (
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+"),
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl)\s+\w+"),
    ),
    "java": (
        re.compile(
            r"^\s*(?:public|protected|private)?\s*(?:abstract\s+)?"
            r"(?:class|interface|enum|record)\s+\w+"
        ),
        re.compile(
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?"
            r"[\w<>?, \[\]]+\s+\w+\s*\([^;]*\)"
        ),
    ),
    "kotlin": (
        re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:data\s+)?(?:class|interface|object|enum\s+class)\s+\w+"),
        re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:suspend\s+)?fun\s+\w+"),
    ),
    "csharp": (
        re.compile(
            r"^\s*(?:public|private|internal|protected)?\s*(?:abstract\s+)?"
            r"(?:class|interface|struct|enum|record)\s+\w+"
        ),
        re.compile(
            r"^\s*(?:public|private|internal|protected)?\s*(?:static\s+)?"
            r"[\w<>?, \[\]]+\s+\w+\s*\([^;]*\)"
        ),
    ),
    "c": (
        re.compile(r"^\s*(?:[\w_*]+\s+)+\w+\s*\([^;]*\)\s*\{"),
        re.compile(r"^\s*(?:typedef\s+)?(?:struct|enum|union)\s+\w+"),
    ),
    "cpp": (
        re.compile(r"^\s*(?:[\w:*&<>]+\s+)+\w+\s*\([^;]*\)\s*\{"),
        re.compile(r"^\s*(?:template\b.*)?(?:class|struct|enum|namespace)\s+\w+"),
    ),
    "ruby": (re.compile(r"^\s*def\s+[\w.?!]+"), re.compile(r"^\s*(?:class|module)\s+\w+")),
    "php": (
        re.compile(r"^\s*(?:(?:public|private|protected|static|abstract)\s+)*function\s+\w+"),
        re.compile(r"^\s*(?:abstract\s+)?(?:class|interface|trait|enum)\s+\w+"),
    ),
    "swift": (
        re.compile(r"^\s*(?:public|private|internal|open|fileprivate)?\s*(?:final\s+)?(?:class|struct|enum|protocol|actor)\s+\w+"),
        re.compile(r"^\s*(?:public|private|internal|open|fileprivate)?\s*func\s+\w+"),
    ),
    "dart": (
        re.compile(r"^\s*(?:abstract\s+)?class\s+\w+"),
        re.compile(r"^\s*(?:Future<[^>]+>|[\w<>?]+)\s+\w+\s*\([^;]*\)\s*\{"),
    ),
    "scala": (
        re.compile(r"^\s*(?:case\s+)?(?:class|object|trait)\s+\w+"),
        re.compile(r"^\s*(?:private\s+|protected\s+)?def\s+\w+"),
    ),
    "elixir": (re.compile(r"^\s*defp?\s+\w+"), re.compile(r"^\s*defmodule\s+[\w.]+")),
    "lua": (re.compile(r"^\s*function\s+[\w.:]+"),),
    "perl": (re.compile(r"^\s*sub\s+\w+"),),
    "bash": (re.compile(r"^\s*(?:function\s+)?\w+\s*\(\)\s*\{"),),
    "sql": (
        re.compile(
            r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?"
            r"(?:TABLE|VIEW|FUNCTION|PROCEDURE)\b",
            re.I,
        ),
    ),
}

_DECLARATION_NODE_KINDS = {
    "function_definition", "function_declaration", "function_item", "method_definition",
    "method_declaration", "method_signature", "class_definition", "class_declaration",
    "struct_item", "struct_declaration", "enum_item", "enum_declaration", "trait_item",
    "interface_declaration", "interface_definition", "impl_item", "type_declaration",
    "type_alias_declaration", "namespace_definition", "namespace_declaration", "module",
    "module_declaration", "object_declaration", "protocol_declaration", "actor_declaration",
    "record_declaration", "constructor_declaration",
}
_IMPORT_NODE_HINTS = ("import", "include", "require", "use_declaration", "using_directive")


@dataclass(frozen=True, slots=True)
class FileOutline:
    """One file's compact, language-neutral contribution to the map.

    The three term sets are the file's searchable form. They are built once at
    parse time and never rendered: ranking a task against a file needs far more
    of the file than the map can afford to show, and comments in particular say
    what a file is *for* while costing nothing to keep out of the prompt.
    """

    path: str
    language: str
    symbols: tuple[str, ...]
    line_count: int
    hidden_symbols: int = 0
    score: float = 0.0
    imports: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    parser: str = "fallback"
    path_terms: frozenset[str] = field(default=frozenset(), repr=False)
    symbol_terms: frozenset[str] = field(default=frozenset(), repr=False)
    text_terms: frozenset[str] = field(default=frozenset(), repr=False)
    """Terms from imports, exports, comments, and docstrings. Weakest evidence,
    and the only place a Chinese task can match at all."""

    def render(self, *, detail: bool = True, max_symbols: int | None = None) -> str:
        """Render a path-only or detailed entry without exposing source bodies."""
        header = f"{self.path} ({self.line_count} lines, {self.language})"
        if not detail:
            return header
        lines = [header]
        if self.imports:
            lines.append("  imports: " + ", ".join(self.imports[:6]))
        if self.exports:
            lines.append("  exports: " + ", ".join(self.exports[:6]))
        symbols = self.symbols if max_symbols is None else self.symbols[:max_symbols]
        lines.extend(f"  {symbol}" for symbol in symbols)
        hidden = self.hidden_symbols + max(len(self.symbols) - len(symbols), 0)
        if hidden:
            lines.append(f"  ... {hidden} more declarations")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RepoMap:
    """The selected map plus accounting and parser diagnostics."""

    text: str
    files_included: int
    files_total: int
    tokens: int
    truncated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    outlines: tuple[FileOutline, ...] = field(default_factory=tuple, repr=False)
    query: str = field(default="", repr=False)
    detailed: tuple[str, ...] = field(default_factory=tuple, repr=False)
    """Paths rendered with their declarations rather than as a bare path. The
    task-ranked layer skips these: repeating a block already in the cached
    system prompt spends the focus budget on nothing."""

    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(slots=True)
class _CachedOutline:
    fingerprint: tuple[int, int]
    outline: FileOutline


class RepoMapIndex:
    """Incrementally parse a workspace and render task-specific selections."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._files: dict[str, _CachedOutline] = {}
        self.scanned = 0
        self.notes: tuple[str, ...] = ()
        self._generation = 0
        self._corpus: _Corpus | None = None

    def refresh(self) -> None:
        """Re-stat the tree and parse only new or changed source files."""
        current: set[str] = set()
        scanned = 0
        notes: list[str] = []
        for root, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for filename in sorted(filenames):
                language = _LANGUAGE_BY_SUFFIX.get(Path(filename).suffix.lower())
                if language is None:
                    continue
                scanned += 1
                if scanned > _MAX_FILES_SCANNED:
                    notes.append(f"stopped scanning after {_MAX_FILES_SCANNED} source files")
                    break
                path = Path(root, filename)
                try:
                    relative = path.relative_to(self.workspace).as_posix()
                    stat = path.stat()
                except OSError:
                    continue
                current.add(relative)
                fingerprint = (stat.st_mtime_ns, stat.st_size)
                cached = self._files.get(relative)
                if cached is None or cached.fingerprint != fingerprint:
                    outline = _outline_file(path, self.workspace, language)
                    if outline is not None:
                        self._files[relative] = _CachedOutline(fingerprint, outline)
                    else:
                        self._files.pop(relative, None)
            if scanned > _MAX_FILES_SCANNED:
                break
        for relative in set(self._files) - current:
            del self._files[relative]
        self.scanned = min(scanned, _MAX_FILES_SCANNED)
        self.notes = tuple(notes)
        self._generation += 1
        self._corpus = None

    @property
    def outlines(self) -> tuple[FileOutline, ...]:
        return tuple(cached.outline for cached in self._files.values())

    def _corpus_stats(self, outlines: Sequence[FileOutline]) -> _Corpus:
        """Document frequencies and vocabulary, rebuilt only after a refresh.

        Both are properties of the tree rather than of the task, so a session
        that asks a dozen questions of an unchanged project pays for them once.
        """
        if self._corpus is not None:
            return self._corpus
        paths: Counter[str] = Counter()
        symbols: Counter[str] = Counter()
        texts: Counter[str] = Counter()
        for outline in outlines:
            paths.update(outline.path_terms)
            symbols.update(outline.symbol_terms)
            texts.update(outline.text_terms)
        self._corpus = _Corpus(
            total=len(outlines),
            path_frequency=paths,
            symbol_frequency=symbols,
            text_frequency=texts,
            vocabulary=tuple(sorted(set(paths) | set(symbols) | set(texts))),
        )
        return self._corpus

    def _ranked(self, query: str) -> list[FileOutline]:
        """Every outline, best first, for ``query`` (or for the tree if empty)."""
        outlines = list(self.outlines)
        if not outlines:
            return outlines
        import_counts = Counter(
            target
            for outline in outlines
            for imported in outline.imports
            for target in _import_targets(imported)
        )
        weights = _weighted_terms(query, self._corpus_stats(outlines))
        return sorted(
            outlines,
            key=lambda outline: (-_relevance(outline, weights, import_counts), outline.path),
        )

    def render(
        self,
        *,
        token_budget: int,
        model: str = "",
        query: str = "",
        layers: Layers = "both",
        abbreviate: Sequence[str] = (),
        limit: int | None = None,
    ) -> RepoMap:
        """Select a map under ``token_budget``.

        Args:
            token_budget: Estimated tokens the rendered text may occupy.
            model: Model name, for choosing a tokeniser.
            query: The task, used to rank files. Empty means rank by structure
                alone, which is what makes a ``paths``/``both`` render stable
                enough to sit in a cached system prompt.
            layers: ``paths`` for the bare index, ``details`` for a ranked
                shortlist, ``both`` for the index with declarations where they
                fit.
            abbreviate: Paths whose declarations are already visible elsewhere.
                They stay in the ranking — the order is the point — but are
                rendered as a bare path rather than repeated in full.
            limit: Cap on how many files a ``details`` render lists.
        """
        if token_budget <= 0:
            return RepoMap("", 0, self.scanned, 0, notes=self.notes, query=query)
        ranked = self._ranked(query)
        if not ranked:
            return RepoMap("", 0, self.scanned, 0, notes=self.notes, query=query)

        if layers == "details":
            return self._render_details(
                ranked[: limit or len(ranked)],
                token_budget,
                model,
                query,
                frozenset(abbreviate),
            )
        return self._render_index(ranked, token_budget, model, query, layers)

    def _render_details(
        self,
        ranked: Sequence[FileOutline],
        token_budget: int,
        model: str,
        query: str,
        abbreviate: frozenset[str],
    ) -> RepoMap:
        """A ranked shortlist, best first.

        No path-only tail beyond ``abbreviate``: this layer answers "which of
        these should I open", and the full index lives elsewhere. Packing stops
        at the first file that will not fit rather than skipping ahead to a
        smaller one, because rank order is the whole product here.
        """
        selected: list[FileOutline] = []
        blocks: list[str] = []
        detailed: list[str] = []
        truncated = False
        tokens = 0
        for outline in ranked:
            if outline.path in abbreviate:
                candidate = outline.render(detail=False)
                if tokens + _cost(candidate, model) > token_budget:
                    truncated = True
                    break
                selected.append(outline)
                blocks.append(candidate)
                tokens += _cost(candidate, model)
                continue

            candidate = outline.render()
            cost = _cost(candidate, model)
            if tokens + cost > token_budget:
                shorter = outline.render(max_symbols=4)
                cost = _cost(shorter, model)
                if tokens + cost > token_budget:
                    truncated = True
                    break
                truncated = truncated or shorter != candidate
                candidate = shorter
            selected.append(outline)
            blocks.append(candidate)
            detailed.append(outline.path)
            tokens += cost

        text = "\n".join(blocks)
        tokens = estimate_text(text, model=model)
        while selected and tokens > token_budget:
            removed = selected.pop()
            blocks.pop()
            if detailed and detailed[-1] == removed.path:
                detailed.pop()
            truncated = True
            text = "\n".join(blocks)
            tokens = estimate_text(text, model=model)
        return RepoMap(
            text=text,
            files_included=len(selected),
            files_total=self.scanned,
            tokens=tokens,
            truncated=truncated or any(outline.hidden_symbols for outline in selected),
            notes=(),
            outlines=tuple(selected),
            query=query,
            detailed=tuple(detailed),
        )

    def _render_index(
        self,
        ranked: Sequence[FileOutline],
        token_budget: int,
        model: str,
        query: str,
        layers: Layers,
    ) -> RepoMap:
        """The path index, with declarations added where the budget allows."""
        # The first layer keeps the tree visible; the second layer spends the
        # remaining budget on the files most likely to answer this task.
        path_budget = (
            token_budget if layers == "paths" else max(min(token_budget // 3, 420), 1)
        )
        selected: list[FileOutline] = []
        details: dict[str, str] = {}
        details_truncated = False
        tokens = 0
        path_tokens = 0
        for outline in ranked:
            minimal = outline.render(detail=False)
            cost = _cost(minimal, model)
            # Always give the highest-ranked file a chance. With a tiny map,
            # one useful path is better than an empty section merely because
            # that path exceeded the one-third path-layer target.
            if selected and path_tokens + cost > path_budget:
                continue
            if tokens + cost > token_budget:
                continue
            selected.append(outline)
            path_tokens += cost
            tokens += cost

        selected_set = {outline.path for outline in selected}
        for outline in ranked if layers == "both" else ():
            if outline.path not in selected_set:
                continue
            minimal = outline.render(detail=False)
            minimal_cost = _cost(minimal, model)
            candidate = outline.render()
            cost = _cost(candidate, model)
            additional = max(cost - minimal_cost, 0)
            if additional > token_budget - tokens:
                full_candidate = candidate
                candidate = outline.render(max_symbols=4)
                cost = _cost(candidate, model)
                additional = max(cost - minimal_cost, 0)
                details_truncated = details_truncated or candidate != full_candidate
            if additional > token_budget - tokens:
                details_truncated = details_truncated or candidate != minimal
                continue
            tokens += additional
            details[outline.path] = candidate

        blocks = [details.get(outline.path, outline.render(detail=False)) for outline in selected]
        text = "\n".join(blocks)
        # Report the estimate for the actual payload. Summing independent
        # blocks is used while selecting because it is cheap, but tokenisers
        # can merge across a newline and make that sum differ slightly.
        tokens = estimate_text(text, model=model)
        while selected and tokens > token_budget:
            removed = selected.pop()
            details.pop(removed.path, None)
            blocks = [details.get(item.path, item.render(detail=False)) for item in selected]
            text = "\n".join(blocks)
            tokens = estimate_text(text, model=model)
        omitted = len(ranked) - len(selected)
        notes = list(self.notes)
        if omitted:
            notes.append(f"{omitted} more files omitted to fit the map budget")
        if any(outline.parser == "fallback" for outline in selected):
            notes.append(
                "some files use fallback declaration matching; install local "
                "Tree-sitter grammars for richer parsing"
            )
        return RepoMap(
            text=text,
            files_included=len(selected),
            files_total=self.scanned,
            tokens=tokens,
            truncated=(
                omitted > 0
                or details_truncated
                or any(outline.hidden_symbols for outline in selected)
            ),
            notes=tuple(dict.fromkeys(notes)),
            outlines=tuple(selected),
            query=query,
            detailed=tuple(details),
        )


def build_repo_map(
    workspace: Path,
    *,
    token_budget: int,
    model: str = "",
    query: str = "",
) -> RepoMap:
    """Build one map, retaining the original public API plus ``query``."""
    index = RepoMapIndex(workspace)
    index.refresh()
    return index.render(token_budget=token_budget, model=model, query=query)


def _outline_file(path: Path, workspace: Path, language: str) -> FileOutline | None:
    try:
        relative = path.relative_to(workspace).as_posix()
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_FILE_BYTES:
        return FileOutline(
            relative,
            language,
            (),
            _count_file_lines(path, size),
            parser="metadata",
            path_terms=frozenset(split_identifier(relative)),
        )
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = source.count("\n") + (1 if source else 0)
    if "\x00" in source:
        return FileOutline(
            relative,
            language,
            (),
            lines,
            parser="binary",
            path_terms=frozenset(split_identifier(relative)),
        )
    symbols, imports, exports, parser = _parse_source(source, language)
    hidden = max(len(symbols) - _MAX_SYMBOLS_PER_FILE, 0)
    prose = prose_text(source, _python_docstrings(source) if language == "python" else ())
    path_terms, symbol_terms, text_terms = _file_terms(
        relative, symbols, imports, exports, prose
    )
    return FileOutline(
        path=relative,
        language=language,
        symbols=tuple(symbols[:_MAX_SYMBOLS_PER_FILE]),
        line_count=lines,
        hidden_symbols=hidden,
        score=_base_score(relative, len(symbols), lines),
        imports=tuple(imports[:12]),
        exports=tuple(exports[:12]),
        parser=parser,
        path_terms=path_terms,
        symbol_terms=symbol_terms,
        text_terms=text_terms,
    )


def _file_terms(
    relative: str,
    symbols: Sequence[str],
    imports: Sequence[str],
    exports: Sequence[str],
    prose: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Split one file into its three searchable fields.

    A term appears in exactly one set, strongest field first, so that scoring
    can stop at the first hit and a name spelled in both the path and a symbol
    is not paid for twice.
    """
    path_terms = split_identifier(relative)

    symbol_terms: set[str] = set()
    for symbol in symbols:
        symbol_terms |= split_identifier(symbol)
        symbol_terms |= cjk_bigrams(symbol)
    symbol_terms -= path_terms

    text_terms: set[str] = set()
    for reference in (*imports, *exports):
        text_terms |= split_identifier(reference)
    text_terms |= split_identifier(prose)
    text_terms |= cjk_bigrams(prose)
    text_terms -= path_terms | symbol_terms
    if len(text_terms) > _MAX_TEXT_TERMS:
        text_terms = set(sorted(text_terms)[:_MAX_TEXT_TERMS])

    return frozenset(path_terms), frozenset(symbol_terms), frozenset(text_terms)


def _parse_source(source: str, language: str) -> tuple[list[str], list[str], list[str], str]:
    if language == "python":
        try:
            return _python_symbols(source), _python_imports(source), [], "python-ast"
        except (SyntaxError, ValueError, RecursionError):
            pass
    tree_result = _tree_sitter_parse(source, language)
    if tree_result is not None:
        return (*tree_result, "tree-sitter")
    return (
        _regex_symbols(source, language),
        _regex_imports(source, language),
        _regex_exports(source, language),
        "fallback",
    )


def _count_file_lines(path: Path, size: int) -> int:
    """Count a large file without decoding or keeping it in memory."""
    if size == 0:
        return 0
    lines = 1
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                lines += chunk.count(b"\n")
    except OSError:
        return 0
    return lines


def _python_symbols(source: str) -> list[str]:
    tree = ast.parse(source)
    symbols: list[str] = []

    def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = _render_arguments(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({args}){returns}"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(signature(node))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            symbols.append(f"class {node.name}({bases})" if bases else f"class {node.name}")
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    if child.name.startswith("_") and child.name != "__init__":
                        continue
                    symbols.append(f"  {signature(child)}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(f"{target.id} = ...")
    return symbols


def _python_docstrings(source: str) -> tuple[str, ...]:
    """First lines of the module, class, and function docstrings.

    A docstring is the one place a file states its purpose in the same register
    a task is written in, and the module docstring in particular is often the
    only text that would match a query phrased in prose rather than in symbols.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ()
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        doc = ast.get_docstring(node)
        if doc:
            found.append(doc.strip().split("\n", 1)[0])
    return tuple(found)


def _python_imports(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return _regex_imports(source, "python")
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _render_arguments(args: ast.arguments) -> str:
    rendered: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    for argument in positional:
        rendered.append(_render_argument(argument))
    if args.vararg:
        rendered.append(f"*{_render_argument(args.vararg)}")
    elif args.kwonlyargs:
        rendered.append("*")
    for argument in args.kwonlyargs:
        rendered.append(_render_argument(argument))
    if args.kwarg:
        rendered.append(f"**{_render_argument(args.kwarg)}")
    return ", ".join(rendered)


def _render_argument(argument: ast.arg) -> str:
    if argument.annotation is None:
        return argument.arg
    try:
        return f"{argument.arg}: {ast.unparse(argument.annotation)}"
    except (AttributeError, ValueError):
        return argument.arg


@lru_cache(maxsize=64)
def _tree_sitter_parser(language: str) -> Any | None:
    """Return a locally cached parser without triggering a grammar download."""
    try:
        import tree_sitter_language_pack as pack
    except ImportError:
        return None
    parser_name = _TREE_SITTER_NAMES.get(language, language)
    downloaded = getattr(pack, "downloaded_languages", None)
    if not callable(downloaded):
        # Older releases do not expose a side-effect-free cache check. Calling
        # get_parser there may download a grammar, so fallback is safer.
        return None
    try:
        if parser_name not in set(downloaded()):
            return None
    except Exception:  # noqa: BLE001 - optional integration must degrade
        return None
    try:
        return pack.get_parser(parser_name)
    except Exception:  # noqa: BLE001 - missing or incompatible grammar
        return None


def _tree_sitter_parse(source: str, language: str) -> tuple[list[str], list[str], list[str]] | None:
    parser = _tree_sitter_parser(language)
    if parser is None:
        return None
    try:
        tree = parser.parse(source.encode("utf-8", "replace"))
        root = tree.root_node
    except Exception:  # noqa: BLE001 - a parser must never break map generation
        return None

    lines = source.splitlines()
    symbols: list[str] = []
    imports: list[str] = []
    exports: list[str] = []
    seen_symbols: set[tuple[int, str]] = set()
    seen_imports: set[str] = set()

    def line_text(node: Any) -> str:
        row = getattr(node, "start_point", (0, 0))[0]
        return lines[row].strip() if 0 <= row < len(lines) else ""

    def visit(node: Any) -> None:
        kind = str(getattr(node, "type", ""))
        line = line_text(node)
        if kind in _DECLARATION_NODE_KINDS:
            name_node = None
            with contextlib.suppress(Exception):
                name_node = node.child_by_field_name("name")
            raw_name = getattr(name_node, "text", b"") or b""
            name = (
                raw_name.decode("utf-8", "replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            if not name and line:
                name_match = re.search(r"\b([A-Za-z_$][\w$]*)\b", line)
                name = name_match.group(1) if name_match else ""
            signature = _clean_tree_signature(line, kind, name)
            key = (int(getattr(node, "start_byte", 0)), signature)
            if signature and key not in seen_symbols:
                seen_symbols.add(key)
                symbols.append(signature)
        if any(hint in kind for hint in _IMPORT_NODE_HINTS) and line and line not in seen_imports:
            seen_imports.add(line)
            imports.append(line)
        if (
            line.startswith(("export ", "pub ", "public "))
            and (kind in _DECLARATION_NODE_KINDS or kind == "export_statement")
            and line not in exports
        ):
            exports.append(line)
        for child in getattr(node, "named_children", ()):
            visit(child)

    visit(root)
    return symbols, imports, exports


def _clean_tree_signature(line: str, kind: str, name: str) -> str:
    rendered = line.strip()
    body = re.search(r"\)\s*\{", rendered)
    if body is not None:
        rendered = rendered[: body.start() + 1]
    elif "{" in rendered:
        rendered = rendered.split("{", 1)[0].rstrip()
    rendered = re.sub(r"\s*=>\s*.*$", "", rendered).strip()
    if len(rendered) > 220:
        rendered = rendered[:217] + "..."
    return rendered or f"{kind.replace('_', ' ')} {name}".strip()


def _regex_symbols(text: str, language: str) -> list[str]:
    if language in {"vue", "astro"}:
        language = "typescript"
    patterns = _REGEX_DECLARATIONS.get(language)
    if patterns is None:
        patterns = (
            re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|fn|func|function)\s+\w+"),
            re.compile(r"^\s*(?:export\s+)?(?:class|struct|interface|trait|enum)\s+\w+"),
        )
    seen: set[str] = set()
    symbols: list[str] = []
    for line in text.split("\n"):
        if len(line) > 400:
            continue
        for pattern in patterns:
            if pattern.match(line) is None:
                continue
            rendered = line.strip().rstrip("{").strip()
            if rendered in seen:
                break
            seen.add(rendered)
            symbols.append(rendered)
            break
    return symbols


def _regex_imports(text: str, language: str) -> list[str]:
    if language in {"vue", "astro"}:
        language = "typescript"
    patterns = {
        "python": re.compile(r"^\s*(?:from\s+([^\s]+)|import\s+([^\s#]+))"),
        "javascript": re.compile(r"^\s*(?:import.*?from\s+|import\s+|require\()(['\"])([^'\"]+)"),
        "typescript": re.compile(r"^\s*(?:import.*?from\s+|import\s+|require\()(['\"])([^'\"]+)"),
        "tsx": re.compile(r"^\s*(?:import.*?from\s+|import\s+|require\()(['\"])([^'\"]+)"),
        "go": re.compile(r"^\s*(?:import\s+)?\"([^\"]+)\""),
        "rust": re.compile(r"^\s*(?:use|mod)\s+([^;]+)"),
        "c": re.compile(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]"),
        "cpp": re.compile(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]"),
        "java": re.compile(r"^\s*import\s+([^;]+)"),
        "kotlin": re.compile(r"^\s*import\s+([^\s]+)"),
        "csharp": re.compile(r"^\s*using\s+([^;]+)"),
        "php": re.compile(r"^\s*(?:use|require|include)\s*[(']?([^';)]+)"),
        "swift": re.compile(r"^\s*import\s+([^\s]+)"),
    }
    pattern = patterns.get(language)
    if pattern is None:
        return []
    imports: list[str] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            imports.append(next((group for group in reversed(match.groups()) if group), ""))
    return imports


def _regex_exports(text: str, language: str) -> list[str]:
    if language in {"vue", "astro"}:
        language = "typescript"
    if language not in {"javascript", "typescript", "tsx"}:
        return []
    return [line.strip() for line in text.splitlines() if line.strip().startswith("export ")]


def _base_score(relative_path: str, symbol_count: int, line_count: int) -> float:
    depth = relative_path.count("/")
    score = 10.0 - depth * 1.5
    score += min(symbol_count, 20) * 0.4
    name = relative_path.rsplit("/", 1)[-1]
    if name in _ENTRY_POINT_NAMES:
        score += 6.0
    if name == "__init__.py":
        score += 1.0 if symbol_count else -2.0
    lowered = relative_path.lower()
    if "test" in lowered or "spec" in lowered:
        score -= 5.0
    if line_count > 1500:
        score -= 1.0
    return score


def _query_terms(query: str) -> list[dict[str, float]]:
    """Weighted lookup terms for one task, grouped by what the task named.

    Three sources, because a task can be phrased three ways. English words and
    identifiers go in at full weight. Chinese goes in twice: as bigrams, which
    match Chinese comments, and as translations, which match the English
    identifiers a Chinese project is actually written in. Without the second,
    a Chinese task matches nothing and the ranking silently becomes the same
    static map for every question.

    Grouped rather than flat because the alternatives within a group are the
    *same* concept — ``page``/``paginate``/``pagination``, or the parts of
    ``OrderService`` — and a file should be paid once for satisfying it, not
    once per spelling. Flattening was enough on its own to float ``errors.py``
    above ``context.py`` for a task about context, on the strength of matching
    both ``error`` and ``errors``.
    """
    groups: list[dict[str, float]] = []

    def group(terms: dict[str, float]) -> None:
        kept = {
            term: weight
            for term, weight in terms.items()
            if not is_stopword(term) and (len(term) >= 2 or term.isdigit())
        }
        if kept:
            groups.append(kept)

    for words in identifier_groups(query.lower()):
        group(dict.fromkeys(words, 1.0))
    for gram in cjk_bigrams(query):
        group({gram: 1.0})
    for translated in expand_cjk(query):
        group(dict.fromkeys(translated, _EXPANSION_WEIGHT))
    return groups


@dataclass(frozen=True, slots=True)
class _Corpus:
    """What the tree as a whole says about how informative a term is.

    Counted per field, not once overall. ``context`` appears in the comments of
    most files here and in exactly one path, so a single frequency would report
    it as uninformative and bury ``context.py`` on a task about contexts. Where
    a term is common is as much the point as how common it is.
    """

    total: int
    path_frequency: Counter[str]
    symbol_frequency: Counter[str]
    text_frequency: Counter[str]
    vocabulary: tuple[str, ...]
    """Every indexed term, sorted, so a prefix range can be found by bisection."""

    def known(self, term: str) -> bool:
        return bool(
            self.path_frequency.get(term)
            or self.symbol_frequency.get(term)
            or self.text_frequency.get(term)
        )


def _idf_scale(frequency: int, total: int) -> float:
    """How much a match should count, given how common the term is in that field.

    ``handler`` in a project of handlers identifies nothing; ``paginate`` in the
    same project identifies one file. Scaled rather than absolute so the field
    weights stay readable, and clamped so a three-file project \u2014 where every
    term looks rare \u2014 does not produce wild scores.
    """
    if total <= 1:
        return 1.0
    rarity = math.log((total + 1) / (frequency + 1)) / math.log(total + 1)
    return _IDF_FLOOR + (_IDF_CEILING - _IDF_FLOOR) * rarity


def _prefix_matches(term: str, corpus: _Corpus) -> tuple[str, ...]:
    """Indexed terms sharing enough of a prefix with ``term`` to be the same word.

    Stands in for a stemmer. Both directions are needed and neither subsumes
    the other: ``cancellation`` extends the indexed ``cancel``, while
    ``pagination`` and the indexed ``paginate`` only share ``paginat``.
    """
    if len(term) < _MIN_PREFIX_LENGTH:
        return ()
    found: list[str] = []

    for length in range(_MIN_PREFIX_LENGTH, len(term)):
        candidate = term[:length]
        if corpus.known(candidate):
            found.append(candidate)

    stem = term[: max(_MIN_PREFIX_LENGTH, len(term) - 3)]
    start = bisect.bisect_left(corpus.vocabulary, stem)
    for candidate in corpus.vocabulary[start : start + 64]:
        if not candidate.startswith(stem):
            break
        if candidate != term and candidate not in found:
            found.append(candidate)
        if len(found) >= _MAX_PREFIX_EXPANSIONS:
            break

    return tuple(found[:_MAX_PREFIX_EXPANSIONS])


def _weighted_terms(query: str, corpus: _Corpus) -> tuple[tuple[_ScoredTerm, ...], ...]:
    """Fold the query, its morphology, and corpus statistics into field weights.

    Everything is resolved here rather than per file, so ranking a thousand
    files is a few set lookups per group rather than a thousand recomputations
    of the same statistics.
    """
    groups = _query_terms(query)
    if not groups:
        return ()

    scored: list[tuple[_ScoredTerm, ...]] = []
    for group in groups:
        resolved: dict[str, float] = dict(group)
        for term, weight in group.items():
            for near in _prefix_matches(term, corpus):
                if near in group:
                    continue
                discounted = weight * _PREFIX_WEIGHT
                resolved[near] = max(resolved.get(near, 0.0), discounted)
        scored.append(
            tuple(
                (
                    term,
                    _PATH_WEIGHT
                    * weight
                    * _idf_scale(corpus.path_frequency.get(term, 0), corpus.total),
                    _SYMBOL_WEIGHT
                    * weight
                    * _idf_scale(corpus.symbol_frequency.get(term, 0), corpus.total),
                    _TEXT_WEIGHT
                    * weight
                    * _idf_scale(corpus.text_frequency.get(term, 0), corpus.total),
                )
                for term, weight in resolved.items()
            )
        )
    return tuple(scored)


def _import_targets(reference: str) -> tuple[str, ...]:
    """Reduce imports from different syntaxes to comparable module names."""
    words = re.findall(r"[A-Za-z_$][\w$]*", reference.lower())
    ignored = {"as", "from", "import", "include", "mod", "require", "use", "using"}
    return tuple(word for word in words if word not in ignored)


def _relevance(
    outline: FileOutline,
    groups: Sequence[Sequence[_ScoredTerm]],
    import_counts: Counter[str],
) -> float:
    """Score one file for one task.

    Matching is against the file's term sets rather than by substring: a
    substring search reports that ``src/history/list.py`` is relevant to a task
    mentioning "is", and short accidental hits like that were outscoring real
    ones.

    Each *group* pays out once, at its best hit, so a file is rewarded for
    covering what the task asked about rather than for how many spellings of it
    happen to occur.
    """
    module_name = Path(outline.path).stem.lower()
    score = outline.score + min(import_counts.get(module_name, 0), 8) * 0.7
    for group in groups:
        best = 0.0
        for term, in_path, in_symbol, in_text in group:
            if term in outline.path_terms:
                best = max(best, in_path)
            elif term in outline.symbol_terms:
                best = max(best, in_symbol)
            elif term in outline.text_terms:
                best = max(best, in_text)
        score += best
    return score


def _collect(workspace: Path) -> tuple[list[FileOutline], int, list[str]]:
    """Compatibility helper retained for callers that used the old internals."""
    index = RepoMapIndex(workspace)
    index.refresh()
    return list(index.outlines), index.scanned, list(index.notes)
