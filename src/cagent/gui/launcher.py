"""Starting the Electron desktop client from the installed CLI.

The desktop app cannot ship inside the Python distribution: its Electron runtime
is around half a gigabyte, so no wheel could carry it. What the CLI *can* do is
find a built checkout and start it with an interpreter that already has
``cagent`` importable — and that is precisely the part a user cannot wire up by
hand, because a ``pipx`` install hides that interpreter in a virtual environment
which is deliberately not on ``PATH``.

So this module resolves three things and hands them to Electron:

* the desktop bundle, from ``CAGENT_DESKTOP``, then ``desktop_path`` in the
  config, then a source checkout sitting next to this package;
* the Electron executable the npm install placed inside that bundle;
* the interpreter to run the JSONL bridge with, which is always this one.

Every failure is a :class:`DesktopLaunchError` carrying the command the user
should run next, because "nothing happened" is the worst possible answer from a
launcher.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import cagent

__all__ = [
    "DesktopLaunchError",
    "electron_binary",
    "find_desktop_bundle",
    "launch_desktop",
]

_BUILD_HINT = "cd {bundle} && npm install && npm run build"


class DesktopLaunchError(RuntimeError):
    """The desktop client could not be located, is unbuilt, or would not start."""


def _import_root() -> Path:
    """The directory that has to be importable for ``import cagent`` to work.

    ``src`` in a checkout, ``site-packages`` in an install. Either way it is what
    the Electron main process needs on ``PYTHONPATH`` before it spawns the bridge.
    """
    return Path(cagent.__file__).resolve().parent.parent


def _candidates(configured: Path | None) -> Iterator[Path]:
    """Where to look for the desktop bundle, most explicit first."""
    from_env = os.environ.get("CAGENT_DESKTOP")
    if from_env:
        yield Path(from_env)
    if configured is not None:
        yield configured
    # A source checkout or an editable install: <repo>/src/cagent -> <repo>/desktop.
    yield _import_root().parent / "desktop"


def find_desktop_bundle(configured: Path | None = None) -> Path:
    """Locate a built desktop bundle, or explain what is missing.

    Args:
        configured: ``desktop_path`` from the config, if the user set it.

    Returns:
        The bundle directory, verified to contain a build the launcher can start.

    Raises:
        DesktopLaunchError: If no candidate is a desktop package, or the one that
            is has not been built yet. These are different problems with
            different fixes, so they get different messages.
    """
    searched: list[Path] = []
    for candidate in _candidates(configured):
        try:
            bundle = candidate.expanduser().resolve()
        except OSError:
            continue
        searched.append(bundle)
        if not (bundle / "package.json").is_file():
            continue
        if not is_built(bundle):
            raise DesktopLaunchError(
                f"The desktop client at {bundle} has not been built.\n"
                f"Run: {_BUILD_HINT.format(bundle=bundle)}"
            )
        return bundle
    looked = "\n".join(f"  {path}" for path in searched) or "  (nowhere)"
    raise DesktopLaunchError(
        "Could not find the desktop client. Looked in:\n"
        f"{looked}\n"
        "Point cagent at a checkout, either with the CAGENT_DESKTOP environment "
        "variable or in ~/.cagent.toml:\n"
        '  [cagent]\n  desktop_path = "/path/to/CodingAgent/desktop"'
    )


def is_built(bundle: Path) -> bool:
    """Whether both halves of the Electron build are present.

    The main process and the renderer are compiled by separate steps, and a
    bundle with only one of them starts into a blank window — which looks like a
    hang rather than a missing build.
    """
    return (bundle / "dist-electron" / "main.js").is_file() and (
        bundle / "dist" / "index.html"
    ).is_file()


def electron_binary(bundle: Path) -> Path:
    """Resolve the Electron executable inside a bundle's ``node_modules``.

    ``electron/path.txt`` is written by the npm package's own install step and
    names the executable for the platform it installed for, so reading it beats
    guessing at per-OS layouts. The guesses remain as a fallback for a bundle
    whose pointer file is missing.

    Raises:
        DesktopLaunchError: If Electron is not installed in this bundle.
    """
    root = bundle / "node_modules" / "electron"
    pointer = root / "path.txt"
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        if name and (root / "dist" / name).is_file():
            return root / "dist" / name
    fallbacks = {
        "win32": root / "dist" / "electron.exe",
        "darwin": root / "dist" / "Electron.app" / "Contents" / "MacOS" / "Electron",
    }
    binary = fallbacks.get(sys.platform, root / "dist" / "electron")
    if binary.is_file():
        return binary
    raise DesktopLaunchError(
        f"Electron is not installed in {bundle}.\n"
        f"Run: {_BUILD_HINT.format(bundle=bundle)}"
    )


def desktop_environment(workspace: Path) -> dict[str, str]:
    """The environment the Electron main process expects.

    ``CAGENT_PYTHON`` is the whole reason this launcher exists. The main process
    falls back to a bare ``python`` on ``PATH``, which for a ``pipx`` install is
    some other interpreter without the agent in it; ``sys.executable`` is by
    definition the one that just imported this module.
    """
    return {
        **os.environ,
        "CAGENT_WORKSPACE": str(workspace),
        "CAGENT_PYTHON": sys.executable,
        "CAGENT_SOURCE_PATH": str(_import_root()),
    }


def launch_desktop(bundle: Path, workspace: Path) -> int:
    """Run the desktop client against ``workspace`` and wait for it to close.

    Returns:
        Electron's exit code, so a failed launch is a failed command.

    Raises:
        DesktopLaunchError: If Electron is missing or the OS refused to start it.
    """
    binary = electron_binary(bundle)
    entry = bundle / "dist-electron" / "main.js"
    try:
        completed = subprocess.run(  # noqa: S603 - both paths are resolved above
            [str(binary), str(entry)],
            cwd=str(workspace),
            env=desktop_environment(workspace),
            check=False,
        )
    except OSError as exc:
        raise DesktopLaunchError(f"Could not start {binary}: {exc}") from exc
    return completed.returncode
