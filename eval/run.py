"""Run the benchmark and report a pass rate.

Usage::

    python -m eval.run                    # every task
    python -m eval.run --task sign-error  # one task
    python -m eval.run --repeat 3         # three attempts each, for variance

Each task runs in a fresh temporary workspace, so a task cannot see another's
files and a failed run leaves nothing behind. Grading runs the task's
verification command in that workspace and reads its exit code; the agent's own
report is recorded for reading but never consulted for the verdict.

Requires a real API key — this is the one part of the project that cannot be
tested with a fake model, because what is being measured *is* the model's
behaviour under this harness.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from cagent.agent.approval import ApprovalPolicy
from cagent.agent.engine import Agent
from cagent.agent.events import CollectingSink, ToolFinished
from cagent.cli.pricing import Price, estimate_cost, parse_prices
from cagent.cli.render import ConsoleRenderer
from cagent.config import AgentConfig, load_config
from cagent.errors import CagentError, ConfigError
from cagent.tools.shell import decode_subprocess_output, kill_process_tree
from cagent.types import Usage

from .tasks import TASKS, Task, task_by_id

VERIFY_TIMEOUT = 120.0


@dataclass(slots=True)
class Attempt:
    """One agent run against one task."""

    task_id: str
    passed: bool
    steps: int
    tools_used: list[str]
    usage: Usage
    elapsed_s: float
    verify_output: str
    reply: str
    error: str | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "passed": self.passed,
            "steps": self.steps,
            "tools": self.tools_used,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "reply": self.reply,
            "verify_output": self.verify_output,
        }


@dataclass(slots=True)
class Report:
    """Every attempt, plus the numbers worth quoting."""

    attempts: list[Attempt] = field(default_factory=list)
    model: str = ""
    prices: dict[str, Price] = field(default_factory=dict)
    """Rates from the user's config, if any; costs are omitted without them."""

    @property
    def passed(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.attempts) if self.attempts else 0.0

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for attempt in self.attempts:
            total = total + attempt.usage
        return total

    def per_task(self) -> dict[str, list[Attempt]]:
        grouped: dict[str, list[Attempt]] = {}
        for attempt in self.attempts:
            grouped.setdefault(attempt.task_id, []).append(attempt)
        return grouped


def materialise(task: Task, root: Path) -> None:
    """Write a task's starting files into ``root``."""
    for name, text in task.files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    for name, data in task.binary_files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def verify(task: Task, root: Path) -> tuple[bool, str]:
    """Run the task's check. Its exit code alone decides the verdict.

    Launched via ``Popen`` and killed as a whole process tree on timeout, rather
    than with ``subprocess.run(timeout=...)``: that kills only the shell, and the
    surviving grandchild keeps the output pipes open, so the wait never returns
    and the benchmark hangs instead of scoring a loss.

    Bytecode writing is disabled for the same reason the shell tool disables it:
    a stale ``.pyc`` can make a fixed file still report the old failure, which
    would show up here as a spurious loss.
    """
    environment = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )

    try:
        process = subprocess.Popen(  # noqa: S602  # the command comes from the task table
            task.verify,
            shell=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            # Its own process group, so the timeout below can kill the tree.
            # Ignored on Windows, where taskkill /T does the same job.
            start_new_session=sys.platform != "win32",
        )
    except OSError as exc:
        return False, f"could not run verification: {exc}"

    try:
        output_bytes, _ = process.communicate(timeout=VERIFY_TIMEOUT)
    except subprocess.TimeoutExpired:
        kill_process_tree(process)
        try:
            output_bytes, _ = process.communicate(timeout=10)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            output_bytes = b""
        output = decode_subprocess_output(output_bytes)
        tail = f"\n{output.strip()}" if output.strip() else ""
        return False, f"verification timed out after {VERIFY_TIMEOUT:g}s{tail}"

    output = decode_subprocess_output(output_bytes)
    return process.returncode == 0, output.strip()


def run_attempt(task: Task, config: AgentConfig, *, verbose: bool) -> Attempt:
    """Set up a workspace, run the agent once, and grade the result."""
    with tempfile.TemporaryDirectory(prefix=f"cagent-eval-{task.id}-") as directory:
        root = Path(directory)
        materialise(task, root)

        attempt_config = _clone_for(config, root, task)
        sink = CollectingSink()
        sinks: list[object] = [sink]
        if verbose:
            sinks.append(ConsoleRenderer(attempt_config, quiet=False))

        from cagent.agent.events import FanOutSink

        fan = FanOutSink(sinks)  # type: ignore[arg-type]
        started = time.monotonic()
        error: str | None = None
        reply = ""

        try:
            agent = Agent.create(
                attempt_config,
                sink=fan,
                # No prompter: the benchmark must not block on a question, and a
                # refusal is a legitimate (failing) outcome rather than a hang.
                policy=ApprovalPolicy(attempt_config, prompter=None),
            )
        except CagentError as exc:
            return Attempt(
                task_id=task.id,
                passed=False,
                steps=0,
                tools_used=[],
                usage=Usage(),
                elapsed_s=0.0,
                verify_output="",
                reply="",
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            result = agent.run_turn(task.prompt)
            reply = result.reply
            if not result.completed:
                error = result.stopped_by
        except CagentError as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            agent.close()

        elapsed = time.monotonic() - started
        passed, output = verify(task, root)

        return Attempt(
            task_id=task.id,
            passed=passed,
            steps=agent.guard.steps,
            tools_used=[event.call.name for event in sink.of_type(ToolFinished)],
            usage=agent.usage,
            elapsed_s=elapsed,
            verify_output=output,
            reply=reply,
            error=error,
        )


def _clone_for(config: AgentConfig, root: Path, task: Task) -> AgentConfig:
    """A copy of the config pointed at this attempt's workspace."""
    import dataclasses

    return dataclasses.replace(
        config,
        workspace=root,
        max_steps=task.max_steps,
        trace_dir=None,
        approval_mode="full-auto",
    )


def print_report(report: Report, *, out: TextIO | None = None) -> None:
    """Print the results table and the summary line.

    ``out`` defaults to whatever ``sys.stdout`` is when this runs, not when the
    module was imported, so redirected output is honoured.
    """
    stream = out if out is not None else sys.stdout

    def write(line: str) -> None:
        print(line, file=stream)

    write("")
    write(f"{'task':<20} {'result':<8} {'steps':>6} {'tokens':>10} {'time':>7}  tools")
    write("-" * 92)

    for task_id, attempts in report.per_task().items():
        for attempt in attempts:
            verdict = "pass" if attempt.passed else "FAIL"
            tools = ", ".join(dict.fromkeys(attempt.tools_used)) or "—"
            write(
                f"{task_id:<20} {verdict:<8} {attempt.steps:>6} "
                f"{attempt.usage.total:>10,} {attempt.elapsed_s:>6.1f}s  {tools[:40]}"
            )
            if attempt.error:
                write(f"{'':<20} └─ stopped: {attempt.error}")
            if not attempt.passed and attempt.verify_output:
                first = attempt.verify_output.strip().split("\n")[-1]
                write(f"{'':<20} └─ check: {first[:70]}")

    write("-" * 92)
    total = len(report.attempts)
    usage = report.total_usage
    write(f"pass@1: {report.passed}/{total} ({report.pass_rate:.0%})")
    write(
        f"tokens: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion"
        + (f" ({usage.cached_tokens:,} cached)" if usage.cached_tokens else "")
    )

    cost = estimate_cost(
        report.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        prices=report.prices,
    )
    if cost is not None:
        write(f"cost:   ${cost:.4f} total, ${cost / max(total, 1):.4f} per attempt")

    steps = [attempt.steps for attempt in report.attempts if attempt.steps]
    if steps:
        write(f"steps:  {statistics.mean(steps):.1f} mean, {max(steps)} worst")
    write(f"model:  {report.model}")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="Run the benchmark tasks and report a pass rate.",
    )
    parser.add_argument("--task", action="append", help="run only this task id (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="attempts per task")
    parser.add_argument("--base-url", metavar="URL", help="the API endpoint")
    parser.add_argument("--model", metavar="NAME", help="the model to benchmark")
    parser.add_argument("--verbose", action="store_true", help="stream each run")
    parser.add_argument("--json", type=Path, help="also write results here as JSON")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    args = parser.parse_args(argv)

    if args.list:
        for task in TASKS:
            tags = f" [{', '.join(task.tags)}]" if task.tags else ""
            print(f"{task.id:<20} {task.prompt[:60]}{tags}")
        return 0

    try:
        selected = [task_by_id(task_id) for task_id in args.task] if args.task else list(TASKS)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    overrides: dict[str, object] = {"base_url": args.base_url, "model": args.model}
    try:
        config = load_config(overrides).validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "The benchmark needs a real endpoint: set base_url, model, and "
            "api_key in .cagent.toml, then try again.",
            file=sys.stderr,
        )
        return 2

    report = Report(
        model=config.resolved_model,
        prices=parse_prices(config.prices),
    )
    total = len(selected) * max(args.repeat, 1)
    done = 0

    for attempt_index in range(max(args.repeat, 1)):
        for task in selected:
            done += 1
            label = f"[{done}/{total}] {task.id}"
            if args.repeat > 1:
                label += f" (attempt {attempt_index + 1})"
            print(f"{label} … ", end="", flush=True)

            attempt = run_attempt(task, config, verbose=args.verbose)
            report.attempts.append(attempt)
            print("pass" if attempt.passed else "FAIL", flush=True)

    print_report(report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": report.model,
                    "pass_rate": report.pass_rate,
                    "passed": report.passed,
                    "total": len(report.attempts),
                    "attempts": [attempt.as_record() for attempt in report.attempts],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0 if report.passed == len(report.attempts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
