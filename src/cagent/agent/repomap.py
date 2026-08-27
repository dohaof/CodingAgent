"""A project skeleton small enough to put in every prompt.

Pasting a repository into the context window is impossible past a few files, and
pasting a file list is nearly useless — names say where code lives but not what
it does. What a programmer actually reads first is the shape: which modules
exist, and what each one declares. So this builds a map of *signatures* only,
ranked by how central each file looks, truncated to a token budget.

The point is not completeness. It is that the model knows a symbol exists and
can then use ``grep_search`` or ``read_file`` to see it properly, instead of
guessing a name or asking for a file it does not need.

Python is parsed with :mod:`ast`, so its signatures are exact — including
decorators, async defs, and base classes. Other languages fall back to
line-anchored regexes, which is imprecise by design: a missed declaration costs
the model one search, while parsing eight languages properly would cost a
dependency this project is not allowed to take.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.tokens import estimate_text
from ..tools.files import IGNORED_DIRS

__all__ = ["FileOutline", "RepoMap", "build_repo_map"]

_MAX_FILES_SCANNED = 2000
"""Ceiling on files examined, so startup stays fast in a huge checkout."""

_MAX_FILE_BYTES = 400_000
"""Files above this are listed but not parsed: generated bundles are large and
declare nothing worth reading."""

_MAX_SYMBOLS_PER_FILE = 24
"""Beyond this a file contributes a count instead of more names, so one large
module cannot crowd out every other file's presence in the map."""

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
    }
)
"""Files that tend to explain a project's entry behaviour."""

_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".sh": "shell",
}

_REGEX_DECLARATIONS: dict[str, tuple[re.Pattern[str], ...]] = {
    "javascript": (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)"),
        re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function"),
    ),
    "go": (
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)"),
        re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface|func)"),
    ),
    "rust": (
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"),
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl)\s+(\w+)"),
    ),
    "java": (
        re.compile(
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?"
            r"(?:class|interface|enum|record)\s+(\w+)"
        ),
    ),
    "c": (re.compile(r"^\s*(?:\w[\w\s*]*?)\b(\w+)\s*\([^;]*\)\s*\{"),),
    "ruby": (
        re.compile(r"^\s*def\s+([\w.?!]+)"),
        re.compile(r"^\s*(?:class|module)\s+(\w+)"),
    ),
    "shell": (re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{"),),
}
_REGEX_DECLARATIONS["typescript"] = _REGEX_DECLARATIONS["javascript"] + (
    re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)"),
)
_REGEX_DECLARATIONS["cpp"] = _REGEX_DECLARATIONS["c"] + (
    re.compile(r"^\s*(?:class|struct|namespace)\s+(\w+)"),
)


@dataclass(frozen=True, slots=True)
class FileOutline:
    """One file's contribution to the map."""

    path: str
    """Workspace-relative, posix-style."""

    language: str
    symbols: tuple[str, ...]
    """Rendered signatures, already indented for nesting."""

    line_count: int
    hidden_symbols: int = 0
    """Declarations omitted by the per-file cap."""

    score: float = 0.0

    def render(self) -> str:
        """The block this file occupies in the map."""
        lines = [f"{self.path} ({self.line_count} lines)"]
        lines.extend(f"  {symbol}" for symbol in self.symbols)
        if self.hidden_symbols:
            lines.append(f"  … {self.hidden_symbols} more declarations")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RepoMap:
    """The rendered skeleton plus what it cost and what it left out."""

    text: str
    files_included: int
    files_total: int
    tokens: int
    truncated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not self.text.strip()


def build_repo_map(
    workspace: Path,
    *,
    token_budget: int,
    model: str = "",
) -> RepoMap:
    """Summarise ``workspace`` into at most ``token_budget`` tokens.

    Args:
        workspace: Project root to scan.
        token_budget: Approximate ceiling on the rendered map.
        model: Model name, for token estimation only.

    Returns:
        The map. Empty when the budget is non-positive or nothing parseable was
        found, which the caller should treat as "omit the section" rather than
        as an error.
    """
    if token_budget <= 0:
        return RepoMap("", 0, 0, 0)

    outlines, scanned, notes = _collect(workspace)
    if not outlines:
        return RepoMap("", 0, scanned, 0, notes=tuple(notes))

    outlines.sort(key=lambda outline: (-outline.score, outline.path))

    blocks: list[str] = []
    tokens = 0
    included = 0
    truncated = False
    for outline in outlines:
        block = outline.render()
        cost = estimate_text(block, model=model)
        if tokens + cost > token_budget:
            truncated = True
            # Keep scanning: a later file may be small enough to still fit, and
            # a partial map ordered by importance is what the budget buys.
            continue
        blocks.append(block)
        tokens += cost
        included += 1

    if truncated:
        notes.append(f"{len(outlines) - included} more files omitted to fit the map budget")

    return RepoMap(
        text="\n".join(blocks),
        files_included=included,
        files_total=scanned,
        tokens=tokens,
        truncated=truncated,
        notes=tuple(notes),
    )


def _collect(workspace: Path) -> tuple[list[FileOutline], int, list[str]]:
    """Outline every supported source file under ``workspace``."""
    outlines: list[FileOutline] = []
    scanned = 0
    notes: list[str] = []

    for root, dirnames, filenames in os.walk(workspace):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for filename in sorted(filenames):
            language = _LANGUAGE_BY_SUFFIX.get(Path(filename).suffix.lower())
            if language is None:
                continue
            scanned += 1
            if scanned > _MAX_FILES_SCANNED:
                notes.append(f"stopped scanning after {_MAX_FILES_SCANNED} source files")
                return outlines, scanned, notes
            path = Path(root, filename)
            outline = _outline_file(path, workspace, language)
            if outline is not None:
                outlines.append(outline)
    return outlines, scanned, notes


def _outline_file(path: Path, workspace: Path, language: str) -> FileOutline | None:
    """Extract one file's declarations, or ``None`` if it is unusable."""
    try:
        size = path.stat().st_size
    except OSError:
        return None

    try:
        relative = path.relative_to(workspace).as_posix()
    except ValueError:
        relative = path.name

    if size > _MAX_FILE_BYTES:
        return FileOutline(relative, language, (), 0, score=0.0)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.count("\n") + 1
    symbols = _python_symbols(text) if language == "python" else _regex_symbols(text, language)
    if not symbols:
        return None

    hidden = max(len(symbols) - _MAX_SYMBOLS_PER_FILE, 0)
    return FileOutline(
        path=relative,
        language=language,
        symbols=tuple(symbols[:_MAX_SYMBOLS_PER_FILE]),
        line_count=lines,
        hidden_symbols=hidden,
        score=_score(relative, len(symbols), lines),
    )


def _python_symbols(source: str) -> list[str]:
    """Exact Python signatures via :mod:`ast`.

    Falls back to the regex path on a syntax error, which is the normal state of
    a file the agent is midway through editing.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return _regex_symbols(source, "python_fallback")

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
                    symbols.append(f"{target.id} = …")
    return symbols


def _render_arguments(args: ast.arguments) -> str:
    """Parameter names with annotations, defaults elided.

    Defaults are dropped because their values are rarely what the model needs
    and can be arbitrarily long expressions.
    """
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


def _regex_symbols(text: str, language: str) -> list[str]:
    """Line-anchored declaration matches for a non-Python language."""
    patterns = _REGEX_DECLARATIONS.get(language)
    if patterns is None:
        patterns = (
            re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),
            re.compile(r"^\s*class\s+(\w+)"),
        )

    seen: set[str] = set()
    symbols: list[str] = []
    for line in text.split("\n"):
        if len(line) > 400:
            continue
        for pattern in patterns:
            match = pattern.match(line)
            if match is None:
                continue
            name = match.group(1)
            if name in seen:
                break
            seen.add(name)
            symbols.append(line.strip().rstrip("{").strip())
            break
    return symbols


def _score(relative_path: str, symbol_count: int, line_count: int) -> float:
    """How likely this file is to matter, for map ordering.

    Shallow files outrank deep ones, declaration-dense files outrank sparse
    ones, entry points and package roots get a bump, and tests are pushed down —
    they are numerous and rarely the thing being asked about, though they stay
    in the map because "where are the tests" is a common question.
    """
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
