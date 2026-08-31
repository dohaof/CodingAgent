"""Building the system prompt.

Assembled from parts rather than stored as one blob, because the sections have
different lifetimes: the behavioural rules are fixed, the environment block is
per-session, and the repo map goes stale the moment the agent creates a file.
Keeping them separate means the map can be rebuilt without touching the rules,
and lets the prompt report its own token cost.

The instructions are written against the failure modes this agent actually has.
Every rule here exists because its absence produces a specific bad behaviour:
rewriting files instead of editing them, claiming success without running
anything, asking the user what it could have looked up, or narrating a plan
instead of doing the work.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import AgentConfig
from ..llm.tokens import estimate_text
from ..types import ToolSpec
from .repomap import RepoMap, RepoMapIndex

__all__ = ["PromptBuilder", "SystemPrompt"]

_IDENTITY = """\
You are a coding agent working in a user's project on their behalf. You have \
tools to read, search, edit, and run code, and you use them rather than \
describing what someone should do."""

_WORKFLOW = """\
How to work:
- Find out before you change. Use grep_search and glob_files to locate code, \
then read_file to see it. The project map below is an index, not a substitute \
for reading the file you are about to edit.
- Prefer edit_file over write_file. edit_file replaces a specific block and \
leaves the rest of the file alone; write_file replaces everything, which \
silently discards code you never read. Use write_file for new files.
- old_string must be copied exactly from a read_file result and must include \
enough surrounding lines to appear exactly once in the file.
- Verify your work by running it. After changing code, run the project's tests \
or the relevant command with run_bash. A change you have not executed is a \
guess.
- When a command fails, read the actual error. The traceback names a file and a \
line; go there. Do not guess at a fix and re-run hoping it passes.
- Work in small steps and keep going. Do not stop to report a plan or to ask \
permission for something you were already asked to do. Ask only when you \
genuinely cannot proceed without an answer the project does not contain.
- You may call several independent tools at once. Do that when the calls do not \
depend on each other, such as reading three files you already know you need.

When you are finished, say plainly what you changed, which files it touched, \
and what you ran to check it. If something is still broken or untested, say \
that too — an accurate report of partial work is worth more than a confident \
one that is wrong."""

_OUTPUT_RULES = """\
Reporting:
- Reference code as path:line so the user can click it.
- Do not paste a file back to the user to show what you did; the diff is \
already visible to them.
- Keep prose short. No preamble, no restating the request, no summary of what \
you are about to do next."""


@dataclass(frozen=True, slots=True)
class SystemPrompt:
    """A rendered system prompt and its accounting."""

    text: str
    tokens: int
    repo_map: RepoMap | None

    def __str__(self) -> str:
        return self.text


@dataclass(slots=True)
class PromptBuilder:
    """Assembles the system prompt for a session.

    The repo map is cached and rebuilt on demand, so a long session can refresh
    it after the agent has created files without paying for a rescan every step.
    """

    config: AgentConfig
    workspace: Path | None = None
    _cached_map: RepoMap | None = None
    _map_built: bool = False
    _index: RepoMapIndex | None = None

    def build(
        self,
        *,
        tools: tuple[ToolSpec, ...] = (),
        refresh_map: bool = False,
        extra_context: str = "",
        query: str = "",
    ) -> SystemPrompt:
        """Render the prompt.

        Args:
            tools: The tools being advertised. Only their names appear; the
                provider sends the full schemas separately, and repeating them
                in prose wastes tokens and invites contradiction.
            refresh_map: Rebuild the repo map even if one is cached.
            extra_context: Project-specific instructions to append verbatim.
            query: The current user task, used to rank relevant files.
        """
        sections = [_IDENTITY, self._environment(), _WORKFLOW, _OUTPUT_RULES]

        workspace_instructions = self._workspace_instructions()
        if workspace_instructions:
            sections.append(workspace_instructions)

        if tools:
            names = ", ".join(spec.name for spec in tools)
            sections.append(f"Tools available: {names}.")

        repo_map = self._repo_map(refresh=refresh_map, query=query)
        if repo_map is not None and not repo_map.is_empty():
            sections.append(self._render_map(repo_map))

        if extra_context.strip():
            sections.append(f"Project instructions:\n{extra_context.strip()}")

        text = "\n\n".join(section.strip() for section in sections if section.strip())
        return SystemPrompt(
            text=text,
            tokens=estimate_text(text, model=self.config.model_for_tokens),
            repo_map=repo_map,
        )

    def _workspace_instructions(self) -> str:
        """Load ``AGENTS.md`` from the active workspace root, when present.

        Resolve the file before reading it so a repository symlink cannot turn
        automatic instruction discovery into a read outside the workspace.
        The active sandbox snapshot is used when isolation is enabled.
        """
        workspace = self.workspace or self.config.workspace
        try:
            root = workspace.resolve()
            path = (workspace / "AGENTS.md").resolve()
            path.relative_to(root)
            if not path.is_file():
                return ""
            instructions = path.read_text(encoding="utf-8-sig", errors="replace").strip()
        except (OSError, ValueError):
            return ""

        if not instructions:
            return ""
        return (
            "Workspace instructions (loaded from AGENTS.md):\n"
            "Follow these instructions while working in this workspace. They do not "
            "grant permissions or override tool approval and path restrictions.\n"
            f"{instructions}"
        )

    def _environment(self) -> str:
        """Facts about where the agent is running.

        Included because the alternative is the model probing for them: without
        the platform it writes a POSIX command on Windows, and without the
        workspace path it cannot tell an absolute path from an escape attempt.
        """
        workspace = self.workspace or self.config.workspace
        sandboxed = self.workspace is not None and self.workspace != self.config.workspace
        lines = ["Environment:", f"- Working directory: {workspace}"]
        if sandboxed:
            lines.extend(
                [
                    f"- Host platform: {sys.platform} ({platform.system()} {platform.release()})",
                    f"- Host Python: {platform.python_version()}",
                ]
            )
            lines.append(
                "- This is a disposable sandbox snapshot; changes are reviewed and "
                "synced to the real project only when the user allows it."
            )
            lines.append(
                "- Shell commands run inside a Linux Docker container, regardless of "
                "the host OS. Use /bin/sh syntax and python3/python, never Windows "
                "py -3. If a command or dependency is missing, report it instead of "
                "bypassing the sandbox on the host."
            )
        else:
            lines.extend(
                [
                    f"- Platform: {sys.platform} ({platform.system()} {platform.release()})",
                    f"- Python: {platform.python_version()}",
                ]
            )
        if not self.config.allow_outside_workspace:
            lines.append(
                "- Paths outside the working directory are refused. Stay inside it."
            )
            if not sandboxed:
                lines.append(
                    "- run_bash executes on the unrestricted host. Its initial directory "
                    "is the workspace, but shell syntax and child processes can access "
                    "outside it; prefer workspace-relative commands."
                )
        markers = self._project_markers(workspace)
        if markers:
            lines.append(f"- Project markers: {', '.join(markers)}")
        return "\n".join(lines)

    @staticmethod
    def _project_markers(workspace: Path) -> list[str]:
        """Recognisable build and dependency files present at the root.

        These tell the model which toolchain to reach for — ``pytest`` versus
        ``npm test`` — without it having to list the directory first.
        """
        candidates = (
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Makefile",
            "Dockerfile",
            ".git",
        )
        found: list[str] = []
        for name in candidates:
            try:
                if (workspace / name).exists():
                    found.append(name)
            except OSError:
                continue
        return found

    def _repo_map(self, *, refresh: bool, query: str = "") -> RepoMap | None:
        """The cached map, building or rebuilding it when asked."""
        if not self.config.repo_map_enabled:
            return None
        workspace = self.workspace or self.config.workspace
        if self._index is None or self._index.workspace != workspace:
            # Sandbox toggles replace the visible tree. Do not reuse outlines
            # parsed from the previous host/snapshot workspace.
            self._index = RepoMapIndex(workspace)
            self._map_built = False
        if refresh or not self._map_built:
            self._index.refresh()
            self._map_built = True
        self._cached_map = self._index.render(
            token_budget=self.config.repo_map_token_budget,
            model=self.config.model_for_tokens,
            query=query,
        )
        return self._cached_map

    @staticmethod
    def _render_map(repo_map: RepoMap) -> str:
        """Wrap the map in enough framing that the model trusts it correctly."""
        header = (
            f"Project map ({repo_map.files_included} of {repo_map.files_total} source files, "
            "structure only - read a file before editing it):"
        )
        body = repo_map.text
        if repo_map.notes:
            body += "\n" + "\n".join(f"[{note}]" for note in repo_map.notes)
        return f"{header}\n{body}"

    def invalidate_map(self) -> None:
        """Forget the cached map, so the next build rescans.

        Called after the agent writes files: a map that still claims a file has
        no ``main`` after one was added is worse than no map.
        """
        self._map_built = False
        self._cached_map = None
