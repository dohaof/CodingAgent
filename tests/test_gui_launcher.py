"""Locating and starting the Electron desktop client from the CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cagent.cli.app import main
from cagent.config import AgentConfig
from cagent.gui import launcher
from cagent.gui.launcher import (
    DesktopLaunchError,
    desktop_environment,
    electron_binary,
    find_desktop_bundle,
    is_built,
)

_REAL_IMPORT_ROOT = launcher._import_root


@pytest.fixture(autouse=True)
def _no_checkout_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the tests from the repository they are running inside.

    The last candidate is a checkout beside the installed package, which in this
    repository is a real built bundle — so a test that expects "not found" would
    instead launch Electron and hang. Point the import root somewhere empty and
    every candidate becomes explicit.
    """
    monkeypatch.setattr(launcher, "_import_root", lambda: tmp_path / "elsewhere" / "src")


def _bundle(root: Path, *, built: bool = True, electron: bool = True) -> Path:
    """A directory shaped like a desktop checkout."""
    bundle = root / "desktop"
    (bundle / "dist-electron").mkdir(parents=True)
    (bundle / "dist").mkdir()
    (bundle / "package.json").write_text('{"name": "cagent-desktop"}', encoding="utf-8")
    if built:
        (bundle / "dist-electron" / "main.js").write_text("//", encoding="utf-8")
        (bundle / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
    if electron:
        dist = bundle / "node_modules" / "electron" / "dist"
        dist.mkdir(parents=True)
        name = "electron.exe" if sys.platform == "win32" else "electron"
        (dist / name).write_text("binary", encoding="utf-8")
        (dist.parent / "path.txt").write_text(name, encoding="utf-8")
    return bundle


def test_the_environment_variable_wins_over_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = _bundle(tmp_path / "chosen")
    ignored = _bundle(tmp_path / "ignored")
    monkeypatch.setenv("CAGENT_DESKTOP", str(chosen))

    assert find_desktop_bundle(configured=ignored) == chosen.resolve()


def test_a_configured_path_is_used_when_no_checkout_sits_beside_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``pipx`` case: the installed package has no repository around it.

    ``site-packages/cagent`` has no ``../../desktop``, so without ``desktop_path``
    there is nothing to find — which is exactly why the setting exists.
    """
    monkeypatch.delenv("CAGENT_DESKTOP", raising=False)
    bundle = _bundle(tmp_path)

    assert find_desktop_bundle(configured=bundle) == bundle.resolve()


def test_an_unbuilt_bundle_reports_the_build_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found-but-unbuilt and not-found-at-all need different instructions."""
    monkeypatch.delenv("CAGENT_DESKTOP", raising=False)
    bundle = _bundle(tmp_path, built=False)

    with pytest.raises(DesktopLaunchError, match="npm run build"):
        find_desktop_bundle(configured=bundle)


def test_a_half_built_bundle_counts_as_unbuilt(tmp_path: Path) -> None:
    """One compile step of two leaves a window that opens onto nothing."""
    bundle = _bundle(tmp_path, built=True)
    (bundle / "dist" / "index.html").unlink()

    assert is_built(bundle) is False


def test_nothing_found_names_every_place_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAGENT_DESKTOP", str(tmp_path / "absent"))

    with pytest.raises(DesktopLaunchError) as caught:
        find_desktop_bundle(configured=tmp_path / "also-absent")

    message = str(caught.value)
    assert "absent" in message and "also-absent" in message
    assert "desktop_path" in message


def test_electron_is_resolved_through_the_pointer_the_npm_install_wrote(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    name = "electron.exe" if sys.platform == "win32" else "electron"

    assert electron_binary(bundle) == bundle / "node_modules" / "electron" / "dist" / name


def test_electron_is_still_found_without_the_pointer_file(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "node_modules" / "electron" / "path.txt").unlink()

    assert electron_binary(bundle).is_file()


def test_a_bundle_without_electron_reports_the_install_command(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, electron=False)

    with pytest.raises(DesktopLaunchError, match="npm install"):
        electron_binary(bundle)


def test_the_desktop_inherits_the_interpreter_that_can_import_cagent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason this launcher exists at all.

    Electron falls back to a bare ``python`` on ``PATH``, which for a ``pipx``
    install is a different interpreter with no agent in it. Passing
    ``sys.executable`` is what makes a global install able to start the backend.
    """
    monkeypatch.setattr(launcher, "_import_root", _REAL_IMPORT_ROOT)

    env = desktop_environment(tmp_path)

    assert env["CAGENT_PYTHON"] == sys.executable
    assert env["CAGENT_WORKSPACE"] == str(tmp_path)
    assert (Path(env["CAGENT_SOURCE_PATH"]) / "cagent").is_dir()


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``main`` with no bundle anywhere and no user config in play.

    A real ``~/.cagent.toml`` with ``desktop_path`` set would otherwise make
    these tests launch Electron and hang, which is a slow way to learn that a
    test read a file it had no business reading.
    """
    monkeypatch.setenv("CAGENT_DESKTOP", str(tmp_path / "no-desktop-here"))
    monkeypatch.setattr(
        "cagent.cli.app.load_config", lambda *args, **kwargs: AgentConfig(workspace=tmp_path)
    )


def _output(capsys: pytest.CaptureFixture[str]) -> str:
    """Captured output with wrapping collapsed, so assertions survive reflow."""
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.usefixtures("cli")
def test_gui_rejects_a_task_argument(capsys: pytest.CaptureFixture[str]) -> None:
    """There is nowhere to put a task: the desktop app has its own composer."""
    assert main(["--gui", "fix the failing tests"]) == 2

    assert "not a task" in _output(capsys)


@pytest.mark.usefixtures("cli")
def test_gui_takes_a_workspace_in_the_task_slot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single existing directory means "open this project"."""
    # Reaching the launcher and failing to find a bundle proves the workspace
    # was accepted; a rejected argument would have stopped at "not a task".
    assert main(["--gui", str(tmp_path)]) == 2
    out = _output(capsys)
    assert "Could not find the desktop client" in out
    # rich reads `[cagent]` as a style tag; unescaped, the instructions would
    # tell the user to write a config key with no table to put it in.
    assert "[cagent]" in out


@pytest.mark.usefixtures("cli")
def test_gui_warns_that_endpoint_flags_are_not_forwarded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silently dropping ``--model`` would look like the flag did not work."""
    main(["--gui", "--model", "some-model", str(tmp_path)])

    assert "not forwarded: model" in _output(capsys)
