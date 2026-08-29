"""Discovering and resolving restorable conversation traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agent.trace import history_from_trace, read_trace
from ..config import AgentConfig

__all__ = [
    "TraceChoice",
    "find_trace_choices",
    "first_user_prompt",
    "resolve_trace_reference",
    "resume_trace_dir",
]


@dataclass(frozen=True, slots=True)
class TraceChoice:
    """A restorable trace and the metadata shown in a conversation picker."""

    path: Path
    session_id: str
    modified: float
    prompt: str
    steps: int
    status: str


def resume_trace_dir(config: AgentConfig) -> Path:
    """Return the directory used by the current workspace's trace writer."""
    return (config.trace_dir or config.workspace / ".cagent" / "traces").expanduser().resolve()


def first_user_prompt(records: list[dict[str, object]]) -> str | None:
    """Return the first non-empty user turn, if the trace has one."""
    for record in records:
        if record.get("type") != "user":
            continue
        text = record.get("text")
        if isinstance(text, str) and text.strip():
            return text
    return None


def find_trace_choices(trace_dir: Path) -> list[TraceChoice]:
    """Scan a trace directory for conversations containing usable history."""
    if not trace_dir.is_dir():
        return []

    choices: list[TraceChoice] = []
    for path in trace_dir.glob("*.jsonl"):
        try:
            records = read_trace(path)
            modified = path.stat().st_mtime
        except OSError:
            continue
        prompt = first_user_prompt(records)
        if not records or not prompt or not history_from_trace(records):
            continue

        session = next((record for record in records if record.get("type") == "session"), {})
        finished = next(
            (
                record
                for record in reversed(records)
                if record.get("type") == "run_finished"
            ),
            {},
        )
        step_records = [record for record in records if record.get("type") == "step_finished"]
        raw_steps = finished.get("steps", len(step_records))
        steps = int(raw_steps) if isinstance(raw_steps, int | float) else len(step_records)
        raw_status = finished.get("reason")
        status = str(raw_status) if isinstance(raw_status, str) else "in progress"
        raw_session = session.get("session_id")
        session_id = raw_session if isinstance(raw_session, str) and raw_session else path.stem
        choices.append(
            TraceChoice(
                path=path,
                session_id=session_id,
                modified=modified,
                prompt=prompt,
                steps=steps,
                status=status,
            )
        )

    return sorted(choices, key=lambda choice: choice.modified, reverse=True)


def resolve_trace_reference(reference: str, trace_dir: Path) -> Path | None:
    """Resolve a full path, filename, or unique session-ID prefix."""
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    name = candidate.name
    names = [name]
    if not name.lower().endswith(".jsonl"):
        names.append(f"{name}.jsonl")
    for item in names:
        exact = trace_dir / item
        if exact.is_file():
            return exact.resolve()

    matches = [path for path in trace_dir.glob("*.jsonl") if path.stem.startswith(name)]
    return matches[0].resolve() if len(matches) == 1 else None
