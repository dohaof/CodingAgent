"""Disposable workspace and Docker execution boundary tests."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from cagent.agent import sandbox as sandbox_module
from cagent.agent.sandbox import SandboxError, SandboxSession
from cagent.config import AgentConfig
from cagent.tools.base import ToolContext
from cagent.tools.shell import _docker_command


def _config(workspace: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(
        workspace=workspace,
        api_key="k",
        sandbox_mode="docker",
        **kwargs,  # type: ignore[arg-type]
    )


def test_docker_is_required_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: False)
    with pytest.raises(SandboxError, match="daemon is unavailable"):
        SandboxSession.create(_config(tmp_path))


def test_auto_falls_back_when_the_local_image_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox_module, "docker_image_available", lambda _image: False)
    config = AgentConfig(workspace=tmp_path, api_key="k", sandbox_mode="auto")

    session, reason = SandboxSession.create_with_status(config)

    assert session is None
    assert reason is not None and "not available locally" in reason


def test_default_sandbox_mode_is_auto(tmp_path: Path) -> None:
    assert AgentConfig(workspace=tmp_path, api_key="k").sandbox_mode == "auto"


def test_snapshot_changes_do_not_touch_real_project(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    try:
        (session.workspace / "app.py").write_text("new\n", encoding="utf-8")
        (session.workspace / "created.txt").write_text("created\n", encoding="utf-8")
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old\n"
        assert session.changed_paths == ("app.py", "created.txt")
        assert "-old" in session.diff() and "+new" in session.diff()
        session.apply()
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"
        assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"
    finally:
        session.close()


def test_snapshot_deletions_are_applied_after_review(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "remove.txt").write_text("remove\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    try:
        (session.workspace / "remove.txt").unlink()
        assert session.changed_paths == ("remove.txt",)
        session.apply()
        assert not (tmp_path / "remove.txt").exists()
    finally:
        session.close()


def test_apply_updates_baseline_for_later_changes(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app.py").write_text("one\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    try:
        (session.workspace / "app.py").write_text("two\n", encoding="utf-8")
        assert session.apply() == ("app.py",)
        assert session.changed_paths == ()
        (session.workspace / "app.py").write_text("three\n", encoding="utf-8")
        assert session.changed_paths == ("app.py",)
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "two\n"
    finally:
        session.close()


def test_discard_changes_resets_snapshot_without_touching_host(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "app.py").write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    try:
        (session.workspace / "app.py").write_text("discard me\n", encoding="utf-8")
        (session.workspace / "new.txt").write_text("discard me\n", encoding="utf-8")
        session.discard_changes()
        assert session.changed_paths == ()
        assert (session.workspace / "app.py").read_text(encoding="utf-8") == "original\n"
        assert not (session.workspace / "new.txt").exists()
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "original\n"
    finally:
        session.close()


def test_sync_refuses_concurrent_host_changes(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    try:
        (session.workspace / "app.py").write_text("sandbox\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("host\n", encoding="utf-8")
        with pytest.raises(SandboxError, match="changed while the sandbox was running"):
            session.apply()
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "host\n"
    finally:
        session.close()


def test_docker_command_has_no_network_and_only_snapshot_mount(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class Snapshot:
        workspace = tmp_path / "snapshot"

    ctx = ToolContext(
        workspace=Snapshot.workspace,
        config=config,
        approve=lambda _request: True,
        emit=lambda _line: None,
        abort=threading.Event(),
        sandbox=Snapshot(),  # type: ignore[arg-type]
        force_workspace_boundary=True,
    )
    argv = _docker_command("pytest -q", ctx, name="cagent-test")
    assert argv[:4] == ["docker", "run", "--rm", "--pull=never"]
    assert argv[4:6] == ["--name", "cagent-test"]
    assert "--network=none" in argv
    assert "--read-only" in argv and "--cap-drop=ALL" in argv
    assert "type=bind,src=" + str(tmp_path / "snapshot") + ",dst=/workspace" in argv
    assert argv[-5:] == ["--entrypoint", "/bin/sh", config.sandbox_image, "-lc", "pytest -q"]


def test_session_container_is_started_once_and_removed_on_close(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    name = session.ensure_container(_config(tmp_path))
    assert session.ensure_container(_config(tmp_path)) == name
    session.close()

    assert len(calls) == 2
    assert calls[0][0:4] == ["docker", "run", "--detach", "--rm"]
    assert calls[1] == ["docker", "rm", "--force", name]


def test_missing_image_has_actionable_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: True)

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            argv,
            125,
            stdout="",
            stderr="docker: Error response from daemon: No such image: local/missing:latest",
        )

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    session = SandboxSession.create(_config(tmp_path))
    assert session is not None
    with pytest.raises(SandboxError, match="docker build -f Dockerfile.agent"):
        session.ensure_container(_config(tmp_path))
    session.close()
