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
import locale
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, ClassVar

from ..agent.sandbox import SandboxError
from ..errors import ToolError, UserAbort
from ..types import RiskLevel
from .base import ApprovalRequest, BaseTool, ToolContext, ToolOutcome
from .schema import Doc
from .truncation import truncate_output

__all__ = [
    "RunBashTool",
    "classify_command",
    "decode_subprocess_output",
    "kill_process_tree",
]

_POSIX = sys.platform != "win32"
_ABORT_POLL_SECONDS = 0.1

_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|passw|credential)")
"""Environment variable names withheld from the child process."""

_SEGMENT_SPLIT = re.compile(r"[\n;]|&&|\|\||\|")
"""Shell separators, so each stage of a compound command is judged on its own."""

_OUTPUT_REDIRECTION = re.compile(r"(?<!<)(?:\d*>>?|&>)")
"""Shell output redirection writes even when the command itself is read-only."""

_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*sudo\b", re.IGNORECASE),
    # File deletion is irreversible from the agent's point of view, even when
    # it lacks a recursive or force flag. Include common POSIX, cmd and
    # PowerShell spellings, plus nested shell invocations.
    re.compile(r"^\s*(rm|rmdir|del|erase|remove-item|ri)\b", re.IGNORECASE),
    re.compile(
        r"^\s*(powershell|pwsh)(\.exe)?\b.*\b(remove-item|del|erase|ri|rm|rmdir)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*cmd(\.exe)?\s+/[ck]\b.*\b(del|erase|rd|rmdir)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfind\b.*\s-delete\b", re.IGNORECASE),
    re.compile(r"^\s*(format|mkfs(\.\w+)?)\b", re.IGNORECASE),
    re.compile(r"^\s*dd\b.*\bif=", re.IGNORECASE),
    re.compile(r"^\s*(shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    re.compile(r"^\s*git\s+push\b.*(--force\b|\s-f\b)", re.IGNORECASE),
    re.compile(r"^\s*git\s+reset\b.*--hard", re.IGNORECASE),
    re.compile(r"^\s*git\s+clean\b.*\s-[a-z]*f", re.IGNORECASE),
    re.compile(r"^\s*chmod\b.*(-R|--recursive).*777", re.IGNORECASE),
    re.compile(r">\s*/dev/(sd|nvme|disk)", re.IGNORECASE),
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
    if _OUTPUT_REDIRECTION.search(text):
        return RiskLevel.MUTATING

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

    Bash is preferred where it can execute in the current OS environment. The
    ``bash.exe`` in Windows System32 is a WSL launcher, not a Windows shell: it
    cannot use the active Windows virtualenv or its paths. Skip that launcher
    and fall back to the platform shell; a real Git Bash remains usable.

    Returns:
        The argv list (or command string) and whether ``shell=True`` is needed.
    """
    if sys.platform == "win32":
        bash = _windows_native_bash()
        if bash:
            return [bash, "-c", command], False
        return command, True

    bash = shutil.which("bash") or "/bin/bash"
    if os.path.exists(bash):
        return [bash, "-c", command], False
    return command, True


def _docker_command(command: str, ctx: ToolContext, *, name: str | None = None) -> list[str]:
    """Build a one-shot constrained Docker invocation.

    Kept as a small, testable helper for callers that need a disposable
    command.  :class:`RunBashTool` uses the session-level container path below
    so normal Agent sessions do not pay a container startup cost per command.
    """
    if ctx.sandbox is None:
        raise ToolError("Docker sandbox is not available for this tool context.")
    image = ctx.config.sandbox_image.strip()
    network = ["--network=bridge"] if ctx.config.sandbox_network else ["--network=none"]
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name or f"cagent-{uuid.uuid4().hex[:12]}",
        *network,
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        str(ctx.config.sandbox_pids),
        "--memory",
        f"{ctx.config.sandbox_memory_mb}m",
        "--cpus",
        str(ctx.config.sandbox_cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--mount",
        # Bind mounts are read/write by default.  Do not append ``rw`` here:
        # it is valid for ``--volume`` but invalid as a ``--mount`` field.
        f"type=bind,src={ctx.workspace},dst=/workspace",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "/bin/sh",
        image,
        # Not a login shell: Debian's /etc/profile assigns PATH outright, which
        # discards the PATH the image's own ENV set. A project image that puts
        # its interpreter on PATH that way -- the usual venv layout -- would see
        # its tools vanish, so a plain -c inherits the container environment.
        "-c",
        command,
    ]


def _docker_exec_command(command: str, container_name: str) -> list[str]:
    """Build a command executed inside an already-running session container."""
    return [
        "docker",
        "exec",
        "--workdir",
        "/workspace",
        container_name,
        "/bin/sh",
        "-c",  # see _docker_command: -l would reset PATH from /etc/profile
        command,
    ]


def _windows_native_bash() -> str | None:
    """Find Git Bash or another native Bash without selecting WSL's launcher."""
    bash = shutil.which("bash")
    if bash and not _is_wsl_launcher(bash):
        return bash

    # Git for Windows may be installed outside PATH while git.exe itself is on
    # PATH through its ``cmd`` directory. Its sibling ``bin`` contains Bash.
    git = shutil.which("git")
    if git:
        root = os.path.dirname(os.path.dirname(os.path.abspath(git)))
        for relative in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
            candidate = os.path.join(root, *relative)
            if os.path.isfile(candidate):
                return candidate
    return None


def _is_wsl_launcher(path: str) -> bool:
    normalized = os.path.normcase(os.path.abspath(path))
    system_root = os.environ.get("SYSTEMROOT")
    if system_root:
        launcher = os.path.normcase(
            os.path.abspath(os.path.join(system_root, "System32", "bash.exe"))
        )
        if normalized == launcher:
            return True
    return "\\windowsapps\\bash.exe" in normalized


def _child_environment() -> dict[str, str]:
    """The environment child processes get.

    Anything whose name looks like a credential is withheld, because the model
    can run ``env`` and whatever it prints lands in the transcript and the trace
    file.

    ``PYTHONDONTWRITEBYTECODE`` is set because this agent's feedback loop is
    "edit, then re-run to verify", and Python decides a cached ``.pyc`` is still
    current from the source's mtime *in whole seconds* plus its size. An edit
    that changes ``a - b`` to ``a + b`` alters neither, so a test re-run within
    the same second can execute the old bytecode and report the bug as
    unfixed — sending the agent off to "fix" code that is already correct.
    Not writing bytecode costs a little import time and makes the verification
    step trustworthy.

    The caller's ``PATH`` is deliberately preserved. A globally installed
    cagent may run from its own pipx environment while working on a project
    whose virtual environment is active in the calling shell. Prepending the
    agent's interpreter directory would silently run tests with the wrong
    Python and dependencies.
    """
    environment = {
        name: value for name, value in os.environ.items() if not _SECRET_PATTERN.search(name)
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Python otherwise inherits the Windows ANSI code page (often GBK), while
    # the same command emits UTF-8 on Unix. Make model-facing output portable.
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def decode_subprocess_output(data: bytes | str | None) -> str:
    """Decode captured command output without assuming the host code page.

    Agent-launched Python is forced to UTF-8, and most modern tools already use
    it. Native Windows programs may still write the active ANSI code page, so
    use that as a fallback after a strict UTF-8 attempt.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data

    encodings = ["utf-8"]
    preferred = locale.getencoding()
    if preferred.lower().replace("-", "") != "utf8":
        encodings.append(preferred)

    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode(encodings[-1], errors="replace")


def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
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


def _communicate_with_watch(
    process: subprocess.Popen[bytes],
    *,
    timeout: float | None,
    abort: threading.Event,
) -> tuple[bytes, bytes, bool, bool]:
    """Drain a process while watching both timeout and user cancellation.

    ``Popen.communicate`` has no abort callback. Running its pipe readers in a
    daemon thread lets the caller poll without blocking, while killing the
    process tree releases inherited pipes before the thread is joined.
    """
    output: dict[str, bytes] = {"stdout": b"", "stderr": b""}
    done = threading.Event()

    def drain() -> None:
        try:
            stdout, stderr = process.communicate()
            output["stdout"] = stdout or b""
            output["stderr"] = stderr or b""
        except (ValueError, OSError):
            pass
        finally:
            done.set()

    threading.Thread(target=drain, name="cagent-process-drain", daemon=True).start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    timed_out = False
    interrupted = False
    while not done.wait(_ABORT_POLL_SECONDS):
        if abort.is_set():
            interrupted = True
            kill_process_tree(process)
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            kill_process_tree(process)
            break

    # Normal completion has already populated the buffers. After a kill, give
    # descendants a bounded moment to release inherited stdout/stderr pipes.
    if not done.is_set():
        done.wait(5.0)
    return output["stdout"], output["stderr"], timed_out, interrupted


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
        "Run a shell command and return its exit code, stdout, and stderr. Shell "
        "execution uses a Docker sandbox when active; otherwise it runs on the "
        "unrestricted host with the current OS user's permissions. Use it to run "
        "tests, builds, and linters: a non-zero exit comes back as output "
        "to read, not as a failure of the call. Long output is truncated in the "
        "middle. Fill in description so the user reading the approval prompt knows "
        "the intent."
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

        argv: list[str] | str
        use_shell: bool
        container_name: str | None = None
        if ctx.sandbox is not None:
            try:
                # One container is shared by all commands in this Agent
                # session.  The snapshot remains the source of truth for
                # file changes and is synchronised only at session close.
                container_name = ctx.sandbox.ensure_container(ctx.config)
            except SandboxError as exc:
                return ToolOutcome.error(
                    str(exc),
                    metadata={"exit_code": None, "timeout": False, "sandbox": "docker"},
                )
            argv, use_shell = _docker_exec_command(command, container_name), False
        else:
            argv, use_shell = _build_invocation(command)
        started = time.perf_counter()
        try:
            process = subprocess.Popen(  # noqa: S603  # running commands is the point
                argv,
                shell=use_shell,
                cwd=str(ctx.workspace) if ctx.sandbox is None else None,
                env=_child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # A process group of its own, so a timeout can kill the whole
                # tree. Ignored on Windows, where taskkill /T does the same job.
                start_new_session=_POSIX,
            )
        except OSError as exc:
            if ctx.sandbox is not None and isinstance(exc, FileNotFoundError):
                return ToolOutcome.error(
                    "Docker sandbox could not start because the Docker CLI is not available. "
                    "Install/start Docker, or set sandbox_mode = 'off'.",
                    metadata={"exit_code": None, "timeout": False, "sandbox": "docker"},
                )
            return ToolOutcome.error(
                f"Could not start the command: {exc}",
                metadata={
                    "exit_code": None,
                    "timeout": False,
                    **({"sandbox": "docker"} if ctx.sandbox is not None else {}),
                },
            )

        stdout_bytes, stderr_bytes, timed_out, interrupted = _communicate_with_watch(
            process, timeout=timeout, abort=ctx.abort
        )
        if (timed_out or interrupted) and container_name is not None and ctx.sandbox is not None:
            # A killed ``docker exec`` client can leave the command running in
            # the container. Recycle it before returning.
            ctx.sandbox.stop_container()
        duration = time.perf_counter() - started

        stdout = decode_subprocess_output(stdout_bytes)
        stderr = decode_subprocess_output(stderr_bytes)
        exit_code = None if timed_out else process.returncode
        if interrupted:
            prefix = "Command interrupted by the user and killed."
        elif timed_out:
            prefix = f"Command timed out after {timeout:g}s and was killed."
        else:
            prefix = ""
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
            "interrupted": interrupted,
            "shell_access": "container" if ctx.sandbox is not None else "host-unrestricted",
        }
        if ctx.sandbox is not None:
            metadata["sandbox"] = "docker"
            if ctx.sandbox.exceeds_size_limit(ctx.config.sandbox_workspace_mb):
                ctx.sandbox.restore()
                return ToolOutcome.error(
                    "The sandbox command exceeded the workspace disk limit. The disposable "
                    "workspace was reset and all pending sandbox changes were discarded.",
                    metadata={
                        **metadata,
                        "sandbox_limit_mb": ctx.config.sandbox_workspace_mb,
                    },
                )
        failed = timed_out or exit_code != 0
        factory = ToolOutcome.error if failed else ToolOutcome.ok
        return factory(body, metadata=metadata, truncated=truncated)

    @staticmethod
    def _drain(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
        """Collect whatever a killed process had already written.

        Partial output is often the most useful part of a timeout — it shows how
        far the command got before hanging.
        """
        try:
            return process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return b"", b""

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
