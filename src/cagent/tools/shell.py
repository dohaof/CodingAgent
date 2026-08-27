"""Running commands: the tool that closes the feedback loop.

Executing the project's own tests and build is what lets the agent find out
whether its edit was right, so this tool exists to bring failure back into the
transcript rather than to succeed quietly. A non-zero exit is not an exception;
it is a result, returned with stderr intact so the model can read the traceback
and fix its own mistake.

Three safeguards bound the blast radius. :func:`classify_command` sorts a
command into a risk tier so read-only work runs unattended while destructive
work reaches the user. A timeout kills the whole process tree, not just the
shell, so a hung child cannot wedge the session. And the child's environment is
stripped of anything that looks like a credential, because the model can run
``env`` and whatever it prints lands in the transcript and the trace log.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Annotated, ClassVar

from ..errors import UserAbort
from ..types import RiskLevel
from .base import ApprovalRequest, BaseTool, ToolContext, ToolOutcome
from .schema import Doc
from .truncation import truncate_output

__all__ = ["RunBashTool", "classify_command", "kill_process_tree"]

_POSIX = sys.platform != "win32"

_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|passw|credential)")
"""Environment variable names withheld from the child process."""

_SEGMENT_SPLIT = re.compile(r"[\n;]|&&|\|\||\|")
"""Shell separators, so each stage of a compound command is judged on its own."""

_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*sudo\b"),
    re.compile(r"^\s*rm\b.*\s-[a-z]*[rf]"),
    re.compile(r"^\s*rm\s+(/|~|\*|\.\.)"),
    re.compile(r"^\s*rmdir\b.*/s"),
    re.compile(r"^\s*del\b.*/[sq]"),
    re.compile(r"^\s*(format|mkfs(\.\w+)?)\b"),
    re.compile(r"^\s*dd\b.*\bif="),
    re.compile(r"^\s*(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"^\s*git\s+push\b.*(--force\b|\s-f\b)"),
    re.compile(r"^\s*git\s+reset\b.*--hard"),
    re.compile(r"^\s*git\s+clean\b.*\s-[a-z]*f"),
    re.compile(r"^\s*chmod\b.*(-R|--recursive).*777"),
    re.compile(r">\s*/dev/(sd|nvme|disk)"),
    re.compile(r"^\s*:\(\)\s*\{.*\}\s*;?\s*:"),
)
"""Irreversible or system-level actions. Deliberately matched on the raw text:
a pattern that fires on a harmless lookalike costs one approval prompt, while a
missed ``rm -rf`` costs the user their working tree."""

_PIPE_TO_SHELL = re.compile(r"(?i)\b(curl|wget)\b[^|]*\|\s*(ba|z|fi|)?sh\b")
"""Downloading a script straight into a shell: whatever runs is unreviewable."""

_SAFE_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "wc",
        "pwd",
        "echo",
        "which",
        "where",
        "whoami",
        "date",
        "find",
        "grep",
        "rg",
        "tree",
        "du",
        "df",
        "ps",
        "true",
        "false",
        "basename",
        "dirname",
        "realpath",
        "stat",
        "file",
        "sort",
        "uniq",
        "cut",
        "diff",
    }
)
"""Commands that only read. Anything absent is treated as mutating, so the
allowlist can be wrong only in the safe direction."""

_SAFE_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "git": frozenset({"status", "log", "diff", "show", "branch", "remote", "blame", "describe"}),
    "pip": frozenset({"list", "show", "freeze"}),
    "pip3": frozenset({"list", "show", "freeze"}),
    "npm": frozenset({"ls", "list", "view"}),
    "cargo": frozenset({"tree"}),
    "go": frozenset({"version", "env", "list"}),
    "docker": frozenset({"ps", "images", "version"}),
}
"""Read-only subcommands of otherwise mutating tools."""

_VERSION_FLAGS = frozenset({"--version", "-v", "-V", "--help", "-h"})
"""Interrogating a tool about itself is always read-only."""

_STDOUT_HEADER = "--- stdout ---"
_STDERR_HEADER = "--- stderr ---"
_SUMMARY_CHARS = 80


def classify_command(command: str) -> RiskLevel:
    """Judge how much damage ``command`` could do.

    This is a heuristic advisory layer, not a sandbox: it decides which calls
    are worth interrupting the user for. The real gate is the approval policy,
    which can require confirmation regardless of what this returns.

    A compound command takes the risk of its riskiest stage, since ``ls &&
    rm -rf build`` is exactly as destructive as the ``rm`` alone.

    Args:
        command: The command line as the model wrote it.

    Returns:
        The highest risk tier any stage of the command falls into.
    """
    if not command.strip():
        return RiskLevel.SAFE
    if _PIPE_TO_SHELL.search(command):
        return RiskLevel.DANGEROUS

    risk = RiskLevel.SAFE
    for segment in _SEGMENT_SPLIT.split(command):
        risk = max(risk, _classify_segment(segment))
    return risk


def _classify_segment(segment: str) -> RiskLevel:
    """Risk of a single command stage."""
    text = segment.strip()
    if not text:
        return RiskLevel.SAFE
    if any(pattern.search(text) for pattern in _DANGEROUS_PATTERNS):
        return RiskLevel.DANGEROUS

    tokens = text.split()
    if not tokens:
        return RiskLevel.SAFE

    head = os.path.basename(tokens[0]).lower()
    head = head.removesuffix(".exe")
    if head in _SAFE_COMMANDS:
        return RiskLevel.SAFE

    rest = [token for token in tokens[1:] if not token.startswith("-")]
    flags = {token for token in tokens[1:] if token.startswith("-")}
    if flags & _VERSION_FLAGS and not rest:
        return RiskLevel.SAFE

    safe_subs = _SAFE_SUBCOMMANDS.get(head)
    if safe_subs is not None and rest and rest[0] in safe_subs:
        return RiskLevel.SAFE

    return RiskLevel.MUTATING


def _build_invocation(command: str) -> tuple[list[str] | str, bool]:
    """Choose how to hand ``command`` to a shell.

    Bash is preferred wherever it exists, including on Windows, so that a model
    writing ordinary POSIX one-liners behaves the same on every machine. Only
    when no bash is found does this fall back to the platform shell.

    Returns:
        The argv list (or command string) and whether ``shell=True`` is needed.
    """
    if sys.platform == "win32":
        bash = shutil.which("bash")
        if bash:
            return [bash, "-c", command], False
        return command, True

    bash = shutil.which("bash") or "/bin/bash"
    if os.path.exists(bash):
        return [bash, "-c", command], False
    return command, True


def _child_environment() -> dict[str, str]:
    """The environment child processes get.

    Two changes from the parent's. Anything whose name looks like a credential
    is withheld, because the model can run ``env`` and whatever it prints lands
    in the transcript and the trace file.

    ``PYTHONDONTWRITEBYTECODE`` is set because this agent's feedback loop is
    "edit, then re-run to verify", and Python decides a cached ``.pyc`` is still
    current from the source's mtime *in whole seconds* plus its size. An edit
    that changes ``a - b`` to ``a + b`` alters neither, so a test re-run within
    the same second can execute the old bytecode and report the bug as
    unfixed — sending the agent off to "fix" code that is already correct.
    Not writing bytecode costs a little import time and makes the verification
    step trustworthy.
    """
    environment = {
        name: value for name, value in os.environ.items() if not _SECRET_PATTERN.search(name)
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill the command and every process it started.

    Killing only the shell would orphan its children, leaving a test runner or
    dev server holding ports and CPU after the tool has returned — and, worse,
    holding the output pipes, so a subsequent ``communicate()`` blocks forever
    on a process nobody is waiting for.

    Exported because anything that imposes a timeout on a shell command needs
    exactly this, and a second copy would be a second chance to get it wrong.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()


def _render(exit_code: int | None, stdout: str, stderr: str, *, prefix: str = "") -> str:
    """Assemble the model-facing report of a finished command."""
    sections: list[str] = []
    if prefix:
        sections.append(prefix)
    sections.append("exit code: " + ("killed" if exit_code is None else str(exit_code)))
    if stdout.strip():
        sections.append(f"{_STDOUT_HEADER}\n{stdout.rstrip()}")
    if stderr.strip():
        sections.append(f"{_STDERR_HEADER}\n{stderr.rstrip()}")
    if len(sections) == (2 if prefix else 1):
        sections.append("(no output)")
    return "\n".join(sections)


@dataclass
class RunBashParams:
    """Arguments for :class:`RunBashTool`."""

    command: Annotated[str, Doc("The shell command to run, in the workspace directory.")]
    timeout: Annotated[
        float | None, Doc("Seconds to allow before the command is killed.")
    ] = None
    description: Annotated[
        str, Doc("One short line on why this command is being run.")
    ] = ""


class RunBashTool(BaseTool):
    """Run a shell command and report its output."""

    name: ClassVar[str] = "run_bash"
    description: ClassVar[str] = (
        "Run a shell command in the workspace and return its exit code, stdout, "
        "and stderr. Use it to run tests, builds, and linters: a non-zero exit "
        "comes back as output to read, not as a failure of the call. Long output "
        "is truncated in the middle. Fill in description so the user reading the "
        "approval prompt knows the intent."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.MUTATING
    Params: ClassVar[type] = RunBashParams

    def run(self, params: RunBashParams, ctx: ToolContext) -> ToolOutcome:
        if ctx.abort.is_set():
            raise UserAbort("Aborted before the command started.")

        command = params.command.strip()
        if not command:
            return ToolOutcome.error("No command was given.")

        timeout = params.timeout if params.timeout and params.timeout > 0 else None
        if timeout is None:
            timeout = ctx.config.bash_timeout

        argv, use_shell = _build_invocation(command)
        started = time.perf_counter()
        try:
            process = subprocess.Popen(  # noqa: S603  # running commands is the point
                argv,
                shell=use_shell,
                cwd=str(ctx.workspace),
                env=_child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                # A process group of its own, so a timeout can kill the whole
                # tree. Ignored on Windows, where taskkill /T does the same job.
                start_new_session=_POSIX,
            )
        except OSError as exc:
            return ToolOutcome.error(
                f"Could not start the command: {exc}",
                metadata={"exit_code": None, "timeout": False},
            )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(process)
            stdout, stderr = self._drain(process)
        duration = time.perf_counter() - started

        exit_code = None if timed_out else process.returncode
        prefix = f"Command timed out after {timeout:g}s and was killed." if timed_out else ""
        body = _render(exit_code, stdout, stderr, prefix=prefix)
        body, truncated = truncate_output(
            body,
            head_lines=ctx.config.tool_output_head_lines,
            tail_lines=ctx.config.tool_output_tail_lines,
            max_chars=ctx.config.tool_output_max_chars,
        )

        metadata: dict[str, object] = {
            "command": command,
            "exit_code": exit_code,
            "duration_s": round(duration, 3),
            "timeout": timed_out,
        }
        failed = timed_out or exit_code != 0
        factory = ToolOutcome.error if failed else ToolOutcome.ok
        return factory(body, metadata=metadata, truncated=truncated)

    @staticmethod
    def _drain(process: subprocess.Popen[str]) -> tuple[str, str]:
        """Collect whatever a killed process had already written.

        Partial output is often the most useful part of a timeout — it shows how
        far the command got before hanging.
        """
        try:
            return process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return "", ""

    def approval_request(
        self, params: RunBashParams, ctx: ToolContext
    ) -> ApprovalRequest | None:
        """Ask for permission unless the command only reads.

        The remembered-approval signature widens with risk in one direction
        only: a mutating command may be blanket-allowed by program name, so
        approving ``pytest`` once covers the next run, while a dangerous command
        signs on its full text and can never be pre-approved by category.
        """
        command = params.command.strip()
        risk = classify_command(command)
        if risk is RiskLevel.SAFE:
            return None

        head = command.split()[0] if command.split() else command
        signature = command if risk is RiskLevel.DANGEROUS else f"run_bash:{head}"
        shown = command if len(command) <= _SUMMARY_CHARS else command[:_SUMMARY_CHARS] + "…"
        summary = f"run: {shown}"
        if params.description.strip():
            summary += f" — {params.description.strip()}"

        detail = f"$ {command}\nin {ctx.rel(ctx.workspace)}"
        if risk is RiskLevel.DANGEROUS:
            detail += "\n\nThis command may be destructive or irreversible."

        return ApprovalRequest(
            tool=self.name,
            risk=risk,
            summary=summary,
            detail=detail,
            signature=signature,
        )

    def preview(self, params: RunBashParams, ctx: ToolContext) -> str | None:
        return f"$ {params.command.strip()}"
