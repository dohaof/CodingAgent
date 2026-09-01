#!/usr/bin/env python3
"""One-shot setup for cagent and its desktop client.

Reached by double-clicking ``install.cmd`` (Windows) or running ``install.sh``
(macOS, Linux), so it assumes a console it can print to and, where a question is
unavoidable, prompt in.

Three things have to line up before ``cagent --gui`` works, and none of them is
obvious from a failure:

* ``cagent`` has to be on ``PATH`` — installed globally, not just importable
  from inside the repository;
* the Electron client has to be installed and compiled, which is a Node build
  the Python packaging cannot perform;
* the CLI has to know where that client is, because a 500 MB Electron runtime
  cannot travel inside a Python wheel.

Every step is idempotent. Re-running after a partial failure — a dropped
download, a missing Node — is the intended way to use this.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESKTOP = REPO / "desktop"
MINIMUM_PYTHON = (3, 11)
MINIMUM_NODE = 18

# The repository's own sources, so this script can use the package it is about to
# install. Everything it touches there is standard-library only, which is what
# makes that safe to do before any dependency exists.
sys.path.insert(0, str(REPO / "src"))


class StepFailed(RuntimeError):
    """A step could not finish. The message says what the user should do."""


def say(message: str = "") -> None:
    print(message, flush=True)


def heading(number: int, total: int, title: str) -> None:
    say()
    say(f"[{number}/{total}] {title}")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, letting its output through, and fail loudly."""
    say(f"    $ {' '.join(command)}")
    try:
        completed = subprocess.run(command, cwd=None if cwd is None else str(cwd), check=False)
    except OSError as exc:
        raise StepFailed(f"Could not run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        raise StepFailed(f"{command[0]} exited with code {completed.returncode}.")


def ask(question: str, *, assume_yes: bool) -> bool:
    """A yes/no prompt that answers itself when there is nobody to ask."""
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        return False
    return input(f"    {question} [Y/n] ").strip().lower() in ("", "y", "yes")


# --------------------------------------------------------------------- checks


def check_python() -> str:
    if sys.version_info < MINIMUM_PYTHON:
        have = ".".join(str(part) for part in sys.version_info[:3])
        want = ".".join(str(part) for part in MINIMUM_PYTHON)
        raise StepFailed(
            f"This is Python {have}; cagent needs {want} or newer.\n"
            "    Install a newer Python and run this again."
        )
    return ".".join(str(part) for part in sys.version_info[:3])


def check_node() -> str:
    """Verify Node is new enough to build the client.

    Reported as a step of its own because the alternative is an npm error
    thirty lines deep in a build log.
    """
    node = shutil.which("node")
    if node is None or shutil.which("npm") is None:
        raise StepFailed(
            "Node.js was not found on PATH; it is needed to build the desktop client.\n"
            f"    Install Node.js {MINIMUM_NODE} or newer from https://nodejs.org/ "
            "and run this again."
        )
    try:
        reported = subprocess.run(
            [node, "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError as exc:
        raise StepFailed(f"Could not run {node}: {exc}") from exc
    major = 0
    if reported.startswith("v") and reported[1:].split(".")[0].isdigit():
        major = int(reported[1:].split(".")[0])
    if major and major < MINIMUM_NODE:
        raise StepFailed(
            f"Node {reported} is too old; the desktop client needs "
            f"{MINIMUM_NODE} or newer."
        )
    return reported or "unknown version"


# ----------------------------------------------------------------- the steps


def _pipx_command(assume_yes: bool) -> list[str]:
    """How to invoke pipx here, installing it first if the user agrees.

    The executable on ``PATH`` is preferred over ``-m pipx`` in this interpreter:
    if the user already has pipx, that is the one holding their other tools, and
    installing cagent anywhere else would leave two copies to keep straight.
    """
    on_path = shutil.which("pipx")
    if on_path is not None:
        return [on_path]
    probe = subprocess.run(
        [sys.executable, "-m", "pipx", "--version"], capture_output=True, check=False
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "pipx"]
    say("    pipx is not installed. It keeps cagent in its own environment,")
    say("    so the agent's dependencies cannot collide with your projects'.")
    if not ask("Install pipx now?", assume_yes=assume_yes):
        raise StepFailed(
            "Skipped. Install pipx yourself and run this again:\n"
            f"      {sys.executable} -m pip install --user pipx\n"
            f"      {sys.executable} -m pipx ensurepath"
        )
    run([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "pipx"])
    run([sys.executable, "-m", "pipx", "ensurepath"])
    return [sys.executable, "-m", "pipx"]


def install_cli(assume_yes: bool) -> None:
    """Put ``cagent`` on PATH, replacing any earlier install of it."""
    pipx = _pipx_command(assume_yes)
    # --force is what makes this idempotent: it reinstalls in place rather than
    # refusing because the package is already there at an older revision.
    run([*pipx, "install", "--force", str(REPO)])


def build_desktop() -> None:
    """Install the client's dependencies and compile both of its halves."""
    npm = shutil.which("npm")
    if npm is None:  # pragma: no cover - check_node already refused this
        raise StepFailed("npm was not found on PATH.")
    if not (DESKTOP / "package.json").is_file():
        raise StepFailed(f"{DESKTOP} does not look like the desktop client.")
    run([npm, "install"], cwd=DESKTOP)
    # `npm install` triggers the package's own prepare script, which builds. Ask
    # again anyway: a bundle left half-built by an interrupted first attempt
    # opens a window onto nothing, which reads as a hang rather than a problem.
    run([npm, "run", "build"], cwd=DESKTOP)
    from cagent.gui.launcher import is_built

    if not is_built(DESKTOP):
        raise StepFailed(
            f"The build finished but {DESKTOP} still has no output.\n"
            "    Check the npm output above for errors."
        )


def record_desktop_path() -> Path:
    """Tell the installed CLI where the client is, for good.

    Written into the config rather than into the environment: it applies to every
    shell immediately, with no terminal to reopen and no per-platform difference
    in how a variable is made to persist.
    """
    from cagent.config import write_setting

    return write_setting("desktop_path", DESKTOP.as_posix())


def verify(*, check_desktop: bool) -> list[str]:
    """Confirm the preconditions actually hold now. Returns any warnings."""
    from cagent.config import load_config
    from cagent.gui.launcher import find_desktop_bundle

    warnings: list[str] = []
    cagent = shutil.which("cagent")
    if cagent is None:
        warnings.append(
            "`cagent` is not on PATH in this shell yet. If pipx was just installed, "
            "open a new terminal; PATH changes do not reach a running one."
        )
    else:
        run([cagent, "--version"])
        # An install predating --gui would pass every other check and then fail
        # at the one command this script exists to enable.
        helped = subprocess.run(
            [cagent, "--help"], capture_output=True, text=True, check=False
        )
        if "--gui" not in helped.stdout:
            warnings.append(
                "the cagent on PATH is older than the --gui flag. Re-run this "
                "without --skip-cli to replace it."
            )
    if check_desktop:
        bundle = find_desktop_bundle(load_config(cwd=REPO).desktop_path)
        say(f"    desktop client: {bundle}")
        if os.environ.get("CAGENT_DESKTOP"):
            warnings.append(
                "CAGENT_DESKTOP is set in your environment and takes precedence over "
                f"the config file: {os.environ['CAGENT_DESKTOP']}"
            )
    return warnings


# ----------------------------------------------------------------------- main


def parser() -> argparse.ArgumentParser:
    at = argparse.ArgumentParser(
        prog="install",
        description="Install cagent and build its desktop client.",
    )
    at.add_argument(
        "-y", "--yes", action="store_true", help="answer every prompt with yes"
    )
    at.add_argument(
        "--skip-cli",
        action="store_true",
        help="leave the existing cagent install alone and only set up the desktop client",
    )
    at.add_argument(
        "--skip-desktop",
        action="store_true",
        help="install the CLI only, without building the desktop client",
    )
    return at


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    say("cagent setup")
    say(f"  repository: {REPO}")

    steps: list[tuple[str, object]] = [("Checking Python", check_python)]
    if not args.skip_desktop:
        steps.append(("Checking Node.js", check_node))
    if not args.skip_cli:
        steps.append(("Installing cagent", lambda: install_cli(args.yes)))
    if not args.skip_desktop:
        steps.append(("Building the desktop client", build_desktop))
        steps.append(("Recording where it lives", record_desktop_path))
    steps.append(("Verifying", lambda: verify(check_desktop=not args.skip_desktop)))

    warnings: list[str] = []
    for number, (title, step) in enumerate(steps, start=1):
        heading(number, len(steps), title)
        try:
            outcome = step()  # type: ignore[operator]
        except StepFailed as exc:
            say()
            say(f"  Stopped: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001 - an installer must not traceback
            say()
            say(f"  Stopped: {type(exc).__name__}: {exc}")
            return 1
        if isinstance(outcome, list):
            warnings.extend(outcome)
        elif outcome is not None:
            say(f"    {outcome}")

    say()
    for warning in warnings:
        say(f"  Note: {warning}")
    say()
    say("  Done. From any directory:")
    say("    cagent            terminal interface")
    say("    cagent --gui      desktop window, using the current directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
