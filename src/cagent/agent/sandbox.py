"""Disposable workspace snapshots for isolated agent runs.

The real project is never mounted into the container.  A sandbox session first
copies the project into a temporary directory; file tools operate on that copy
and :class:`~cagent.tools.shell.RunBashTool` mounts the copy read/write inside a
Docker container.  At the end of a run the copy is either discarded or merged
back after a conflict check and an explicit approval request.

Docker is an execution boundary, not a complete security proof.  The container
is therefore started with no network by default (or the default bridge network
when explicitly enabled), a read-only root filesystem, no Linux capabilities,
a temporary ``/tmp``, and finite CPU/memory/process limits.  The Docker socket
is deliberately never mounted.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

from ..config import AgentConfig
from ..errors import CagentError

__all__ = [
    "SandboxError",
    "SandboxSession",
    "docker_available",
    "docker_image_available",
]

_MAX_DIFF_FILE_BYTES = 200_000
_MAX_DIFF_CHARS = 24_000
_COPY_IGNORED_ROOTS = frozenset(
    {
        ".cagent",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
)
_IGNORED_SYNC_ROOTS = frozenset(
    {
        ".cagent",
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
)


def _tree_size(root: Path) -> int:
    """Return the regular-file bytes in a snapshot without following links."""
    total = 0
    for current, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(current) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


class SandboxError(CagentError):
    """The disposable workspace could not be created or synchronised."""


@dataclass(frozen=True, slots=True)
class _Entry:
    kind: str
    digest: str = ""
    target: str = ""
    mode: int = 0


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_ignored(relative: str) -> bool:
    # ``copytree`` excludes these directories at every depth.  The manifest
    # must apply the same rule, otherwise a nested desktop/node_modules or
    # package/dist appears as a deletion when the disposable snapshot closes.
    parts = relative.split("/")
    return any(part in _IGNORED_SYNC_ROOTS or _is_secret_file(part) for part in parts)


def _is_secret_file(name: str) -> bool:
    return name == ".git-credentials" or name == ".cagent.toml" or name.startswith(
        ".cagent.toml."
    )


def _copy_ignore(_path: str, names: list[str]) -> list[str]:
    return [
        name for name in names if name in _COPY_IGNORED_ROOTS or _is_secret_file(name)
    ]


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise SandboxError(
            f"Could not read {path} while building a sandbox manifest: {exc}"
        ) from exc
    return hasher.hexdigest()


def _manifest(root: Path) -> dict[str, _Entry]:
    """Describe regular files, symlinks, and directories without following links."""
    result: dict[str, _Entry] = {}
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(dirnames)
        filenames = sorted(filenames)
        for name in dirnames + filenames:
            path = current_path / name
            relative = _relative(path, root)
            if _is_ignored(relative):
                if name in dirnames:
                    dirnames.remove(name)
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                raise SandboxError(f"Could not inspect {path}: {exc}") from exc
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                result[relative] = _Entry("symlink", target=os.readlink(path), mode=mode)
            elif stat.S_ISDIR(info.st_mode):
                result[relative] = _Entry("directory", mode=mode)
            elif stat.S_ISREG(info.st_mode):
                result[relative] = _Entry("file", digest=_digest(path), mode=mode)
            else:
                # Device files, sockets, and FIFOs are not copied back.  They
                # are outside the source tree a coding agent should modify.
                continue
    return result


def _copytree(source: Path, destination: Path) -> None:
    """Copy a project while leaving trace files out of the disposable tree."""
    try:
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=_copy_ignore,
        )
    except OSError as exc:
        raise SandboxError(f"Could not create sandbox snapshot of {source}: {exc}") from exc

    # Git metadata is useful for status/diff, but a remote URL can contain an
    # embedded token. Runtime networking is disabled; redact it anyway so a
    # model cannot print the credential into its transcript.
    git_config = destination / ".git" / "config"
    try:
        text = git_config.read_text(encoding="utf-8")
        scrubbed = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1redacted@", text)
        if scrubbed != text:
            git_config.write_text(scrubbed, encoding="utf-8")
    except OSError:
        pass


def _validate_symlinks(
    root: Path, manifest: dict[str, _Entry], paths: set[str] | None = None
) -> None:
    """Reject links in a snapshot that point outside the snapshot root."""
    for relative, entry in manifest.items():
        if paths is not None and relative not in paths:
            continue
        if entry.kind != "symlink":
            continue
        link = root / Path(relative)
        target = (link.parent / entry.target).resolve()
        if target != root and not target.is_relative_to(root):
            raise SandboxError(
                f"Refusing to synchronise symlink {relative!r}: target escapes the workspace."
            )


def docker_available() -> bool:
    """Return whether the Docker CLI and daemon answer a read-only probe."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def docker_image_available(image: str) -> bool:
    """Return whether ``image`` exists locally without pulling it."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _remove(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SandboxError(f"Could not inspect {path}: {exc}") from exc
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SandboxError(f"Could not create parent directory for {path}: {exc}") from exc


def _ensure_safe_parent(root: Path, relative: str) -> None:
    """Ensure a host-side sync cannot traverse a pre-existing symlink."""
    parent = root / Path(relative).parent
    current = root
    try:
        parts = parent.relative_to(root).parts
    except ValueError as exc:
        raise SandboxError(f"Refusing to synchronise path outside the project: {relative}") from exc
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            # Missing parents are safe: _ensure_parent creates them below.
            continue
        except OSError as exc:
            raise SandboxError(f"Could not inspect sync parent {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise SandboxError(
                f"Refusing to synchronise {relative!r}: its host parent is not a real directory."
            )


@dataclass(slots=True)
class SandboxSession:
    """A temporary project copy and its baseline for conflict-safe syncing."""

    real_workspace: Path
    workspace: Path
    baseline: dict[str, _Entry]
    container_name: str | None = None
    _closed: bool = False

    @classmethod
    def create(cls, config: AgentConfig) -> SandboxSession | None:
        """Create the selected snapshot, discarding automatic fallback detail."""
        session, _fallback_reason = cls.create_with_status(config)
        return session

    @classmethod
    def create_with_status(
        cls, config: AgentConfig
    ) -> tuple[SandboxSession | None, str | None]:
        """Create a snapshot and report why ``auto`` selected the host shell."""
        if config.sandbox_mode == "off":
            return None, "Docker sandboxing was explicitly disabled."
        if config.sandbox_mode not in ("auto", "docker"):
            raise SandboxError(f"Unsupported sandbox mode: {config.sandbox_mode!r}")
        if config.allow_outside_workspace:
            if config.sandbox_mode == "docker":
                raise SandboxError(
                    "allow_outside_workspace cannot be combined with Docker sandboxing."
                )
            return (
                None,
                "Automatic Docker isolation was skipped because "
                "allow_outside_workspace = true requests unrestricted file access.",
            )
        if not docker_available():
            detail = "The Docker CLI or daemon is unavailable."
            if config.sandbox_mode == "auto":
                return None, detail
            raise SandboxError(
                "Docker sandbox requested, but the Docker daemon is unavailable. "
                "Start Docker Desktop/Engine or set sandbox_mode = 'off'."
            )
        if not docker_image_available(config.sandbox_image.strip()):
            detail = (
                f"Sandbox image {config.sandbox_image!r} is not available locally; "
                "cagent never pulls images automatically."
            )
            if config.sandbox_mode == "auto":
                return None, detail
            # Forced isolation must fail before a snapshot exists. Creating one
            # anyway leaves the file tools writing to a copy that ``run_bash``
            # can never execute in: the agent could edit but never verify.
            raise SandboxError(
                f"Docker sandbox requested, but image {config.sandbox_image!r} is not "
                "available locally. Build or pull it first, or set sandbox_mode = 'off'."
            )
        real = config.workspace
        if not real.is_dir():
            raise SandboxError(f"Workspace is not a directory: {real}")
        baseline = _manifest(real)
        try:
            parent = Path(tempfile.mkdtemp(prefix="cagent-sandbox-"))
        except OSError as exc:
            raise SandboxError(f"Could not allocate a temporary sandbox directory: {exc}") from exc
        snapshot = parent / "workspace"
        try:
            _copytree(real, snapshot)
        except Exception:
            shutil.rmtree(parent, ignore_errors=True)
            raise
        session = cls(real_workspace=real, workspace=snapshot, baseline=baseline)
        if session.exceeds_size_limit(config.sandbox_workspace_mb):
            session.close()
            raise SandboxError(
                f"Workspace snapshot exceeds sandbox_workspace_mb={config.sandbox_workspace_mb}."
            )
        return session, None

    def ensure_container(self, config: AgentConfig) -> str:
        """Start the session container once and return its stable name.

        The project snapshot is persistent for the lifetime of an Agent.  A
        long-lived container means repeated ``run_bash`` calls reuse the same
        image layers and any packages installed during the session.  The
        container still has no durable writable layer: only ``/workspace``
        and the bounded ``/tmp`` tmpfs are writable, and :meth:`close` removes
        it with ``--rm``.
        """
        if self._closed:
            raise SandboxError("The sandbox session is already closed.")
        if self.container_name is not None:
            return self.container_name

        name = f"cagent-{uuid.uuid4().hex[:12]}"
        image = config.sandbox_image.strip()
        network = ["--network=bridge"] if config.sandbox_network else ["--network=none"]
        argv = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--name",
            name,
            *network,
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit",
            str(config.sandbox_pids),
            "--memory",
            f"{config.sandbox_memory_mb}m",
            "--cpus",
            str(config.sandbox_cpus),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--mount",
            # Bind mounts are read/write by default; ``rw`` is a valid
            # ``--volume`` option but not a key accepted by ``--mount``.
            f"type=bind,src={self.workspace},dst=/workspace",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "while :; do sleep 3600; done",
        ]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"Could not start the Docker sandbox container: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if "No such image" in detail or "pull access denied" in detail.lower():
                raise SandboxError(
                    f"Sandbox image {image!r} is not available locally. "
                    f"Build it locally (e.g. 'docker build -t {image} .') "
                    f"or pull it explicitly with 'docker pull {image}'."
                )
            if len(detail) > 2_000:
                detail = detail[:2_000] + "..."
            raise SandboxError(
                "Could not start the Docker sandbox container"
                + (f": {detail}" if detail else ".")
            )
        self.container_name = name
        # Normal interpreter shutdown (including an unhandled exception) must
        # not leave a detached sandbox consuming resources. SIGKILL/power loss
        # cannot be handled, but the container can be removed manually with
        # ``docker ps --filter name=cagent-``.
        atexit.register(self.stop_container)
        return name

    def stop_container(self) -> None:
        """Force-remove the session container, if one was started."""
        name = self.container_name
        self.container_name = None
        if name is None:
            return
        with suppress(Exception):
            atexit.unregister(self.stop_container)
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                check=False,
                timeout=10,
            )

    @property
    def changed_paths(self) -> tuple[str, ...]:
        """Workspace-relative paths changed since the snapshot was created."""
        current = _manifest(self.workspace)
        paths = sorted(set(self.baseline) | set(current))
        changed = tuple(path for path in paths if self.baseline.get(path) != current.get(path))
        _validate_symlinks(self.workspace, current, set(changed))
        return changed

    def exceeds_size_limit(self, limit_mb: int) -> bool:
        """Whether the disposable copy exceeded its configured disk budget."""
        return _tree_size(self.workspace) > limit_mb * 1024 * 1024

    def restore(self) -> None:
        """Reset a damaged or oversized disposable copy to its original baseline."""
        # A bind mount follows the old directory object even if its host path
        # is atomically replaced below. Recycle the container first so the
        # next command mounts the restored snapshot.
        self.stop_container()
        replacement = self.workspace.parent / "workspace-restored"
        _remove(replacement)
        _copytree(self.real_workspace, replacement)
        old = self.workspace.parent / "workspace-old"
        _remove(old)
        os.replace(self.workspace, old)
        os.replace(replacement, self.workspace)
        _remove(old)

    def diff(self) -> str:
        """Render a bounded human-readable diff of the pending snapshot changes."""
        current = _manifest(self.workspace)
        changed = [
            path
            for path in sorted(set(self.baseline) | set(current))
            if self.baseline.get(path) != current.get(path)
        ]
        _validate_symlinks(self.workspace, current, set(changed))
        chunks: list[str] = []
        for relative in changed:
            before = self.baseline.get(relative)
            after = current.get(relative)
            source = self.workspace / Path(relative)
            target = self.real_workspace / Path(relative)
            if before and after and before.kind == after.kind == "file":
                try:
                    small_enough = (
                        source.stat().st_size <= _MAX_DIFF_FILE_BYTES
                        and target.stat().st_size <= _MAX_DIFF_FILE_BYTES
                    )
                    if small_enough:
                        old = target.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines(keepends=True)
                        new = source.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines(keepends=True)
                        chunks.extend(
                            unified_diff(
                                old,
                                new,
                                fromfile=f"a/{relative}",
                                tofile=f"b/{relative}",
                            )
                        )
                        continue
                except OSError:
                    pass
            if before is None:
                chunks.append(f"[added {relative} ({after.kind if after else 'unknown'})]\n")
            elif after is None:
                chunks.append(f"[deleted {relative} ({before.kind})]\n")
            else:
                chunks.append(f"[changed {relative}: {before.kind} -> {after.kind}]\n")
        text = "".join(chunks)
        if len(text) > _MAX_DIFF_CHARS:
            omitted = len(text) - _MAX_DIFF_CHARS
            text = text[:_MAX_DIFF_CHARS] + f"\n... {omitted} diff characters omitted"
        return text or "(sandbox produced no project changes)"

    def apply(self) -> tuple[str, ...]:
        """Merge snapshot changes back after detecting concurrent host edits."""
        if _manifest(self.real_workspace) != self.baseline:
            raise SandboxError(
                "The real workspace changed while the sandbox was running; "
                "changes were not copied back. Review the sandbox diff and retry."
            )

        current = _manifest(self.workspace)
        changed = [
            path
            for path in sorted(set(self.baseline) | set(current))
            if self.baseline.get(path) != current.get(path)
        ]
        _validate_symlinks(self.workspace, current, set(changed))
        removals = [path for path in changed if path not in current]
        additions = [path for path in changed if path in current]

        # Remove deepest paths first, then create directories before files.
        for relative in sorted(removals, key=lambda item: (item.count("/"), item), reverse=True):
            _ensure_safe_parent(self.real_workspace, relative)
            _remove(self.real_workspace / Path(relative))
        for relative in sorted(additions, key=lambda item: (item.count("/"), item)):
            _ensure_safe_parent(self.real_workspace, relative)
            self._copy_entry(self.workspace / Path(relative), self.real_workspace / Path(relative))
        # The applied snapshot is now the new conflict-detection baseline.
        # This makes an explicit mid-session ``/sandbox apply`` a real commit:
        # later changes are compared only with the state just synced.
        self.baseline = current
        return tuple(changed)

    def discard_changes(self) -> None:
        """Discard unsynchronised edits and reset the snapshot from the host."""
        self.restore()
        # If the host changed concurrently, those host edits are intentionally
        # adopted as the new baseline rather than copied back or overwritten.
        self.baseline = _manifest(self.workspace)

    @staticmethod
    def _copy_entry(source: Path, destination: Path) -> None:
        try:
            info = source.lstat()
        except OSError as exc:
            raise SandboxError(f"Could not inspect sandbox path {source}: {exc}") from exc
        _ensure_parent(destination)
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            if destination.exists() and not destination.is_dir():
                _remove(destination)
            destination.mkdir(exist_ok=True)
            with suppress(OSError):
                os.chmod(destination, stat.S_IMODE(info.st_mode))
            return
        _remove(destination)
        if stat.S_ISLNK(info.st_mode):
            try:
                temporary = destination.with_name(destination.name + ".cagent-link-tmp")
                _remove(temporary)
                os.symlink(os.readlink(source), temporary)
                os.replace(temporary, destination)
            except OSError as exc:
                raise SandboxError(f"Could not synchronise symlink {destination}: {exc}") from exc
            return
        if stat.S_ISREG(info.st_mode):
            temporary_fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.cagent-", dir=str(destination.parent)
            )
            os.close(temporary_fd)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            except OSError as exc:
                _remove(temporary)
                raise SandboxError(f"Could not synchronise file {destination}: {exc}") from exc
            return
        raise SandboxError(f"Refusing to synchronise unsupported file type: {source}")

    def close(self) -> None:
        """Delete the temporary snapshot; the real project is left untouched."""
        if self._closed:
            return
        self._closed = True
        self.stop_container()
        with suppress(OSError):
            shutil.rmtree(self.workspace.parent)
