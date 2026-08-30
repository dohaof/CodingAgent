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
import contextlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..llm.tokens import estimate_text
from ..tools.files import IGNORED_DIRS

__all__ = ["FileOutline", "RepoMap", "RepoMapIndex", "build_repo_map"]

_MAX_FILES_SCANNED = 2000
_MAX_FILE_BYTES = 400_000
_MAX_SYMBOLS_PER_FILE = 24

_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "change",
        "fix",
        "for",
        "implement",
        "in",
        "of",
        "or",
        "please",
        "the",
        "to",
        "update",
    }
)

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
    """One file's compact, language-neutral contribution to the map."""

    path: str
    language: str
    symbols: tuple[str, ...]
    line_count: int
    hidden_symbols: int = 0
    score: float = 0.0
    imports: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    parser: str = "fallback"

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

    @property
    def outlines(self) -> tuple[FileOutline, ...]:
        return tuple(cached.outline for cached in self._files.values())

    def render(self, *, token_budget: int, model: str = "", query: str = "") -> RepoMap:
        """Select a diverse, task-relevant map under ``token_budget``."""
        if token_budget <= 0:
            return RepoMap("", 0, self.scanned, 0, notes=self.notes, query=query)
        outlines = list(self.outlines)
        if not outlines:
            return RepoMap("", 0, self.scanned, 0, notes=self.notes, query=query)

        terms = _query_terms(query)
        import_counts = Counter(
            target
            for outline in outlines
            for imported in outline.imports
            for target in _import_targets(imported)
        )
        ranked = sorted(
            outlines,
            key=lambda outline: (-_relevance(outline, terms, import_counts), outline.path),
        )

        # The first layer keeps the tree visible; the second layer spends the
        # remaining budget on the files most likely to answer this task.
        path_budget = max(min(token_budget // 3, 420), 1)
        selected: list[FileOutline] = []
        details: dict[str, str] = {}
        details_truncated = False
        tokens = 0
        path_tokens = 0
        for outline in ranked:
            minimal = outline.render(detail=False)
            cost = estimate_text(minimal, model=model)
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
        for outline in ranked:
            if outline.path not in selected_set:
                continue
            minimal = outline.render(detail=False)
            minimal_cost = estimate_text(minimal, model=model)
            candidate = outline.render()
            cost = estimate_text(candidate, model=model)
            additional = max(cost - minimal_cost, 0)
            if additional > token_budget - tokens:
                full_candidate = candidate
                candidate = outline.render(max_symbols=4)
                cost = estimate_text(candidate, model=model)
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
        omitted = len(outlines) - len(selected)
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
        )
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = source.count("\n") + (1 if source else 0)
    if "\x00" in source:
        return FileOutline(relative, language, (), lines, parser="binary")
    symbols, imports, exports, parser = _parse_source(source, language)
    hidden = max(len(symbols) - _MAX_SYMBOLS_PER_FILE, 0)
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
    )


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


def _query_terms(query: str) -> tuple[str, ...]:
    terms: set[str] = set()
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+|[\u3400-\u9fff]{2,}", query.lower())
    for raw in raw_terms:
        candidates = {raw}
        candidates.update(part for part in raw.split("_") if part)
        candidates.update(part.lower() for part in re.findall(r"[a-z]{2,}|[0-9]+", raw))
        terms.update(
            candidate
            for candidate in candidates
            if candidate not in _QUERY_STOPWORDS
            and (len(candidate) >= 2 or candidate.isdigit())
        )
    return tuple(sorted(terms))


def _import_targets(reference: str) -> tuple[str, ...]:
    """Reduce imports from different syntaxes to comparable module names."""
    words = re.findall(r"[A-Za-z_$][\w$]*", reference.lower())
    ignored = {"as", "from", "import", "include", "mod", "require", "use", "using"}
    return tuple(word for word in words if word not in ignored)


def _relevance(
    outline: FileOutline,
    terms: tuple[str, ...],
    import_counts: Counter[str],
) -> float:
    module_name = Path(outline.path).stem.lower()
    score = outline.score + min(import_counts.get(module_name, 0), 8) * 0.7
    if not terms:
        return score
    haystack = " ".join(
        (outline.path, *outline.symbols, *outline.imports, *outline.exports)
    ).lower()
    for term in terms:
        if term in outline.path.lower():
            score += 12.0
        if any(term in symbol.lower() for symbol in outline.symbols):
            score += 9.0
        if term in haystack:
            score += 2.0
    return score


def _collect(workspace: Path) -> tuple[list[FileOutline], int, list[str]]:
    """Compatibility helper retained for callers that used the old internals."""
    index = RepoMapIndex(workspace)
    index.refresh()
    return list(index.outlines), index.scanned, list(index.notes)
