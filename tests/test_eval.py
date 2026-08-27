"""The benchmark harness itself.

A benchmark that silently mismeasures is worse than none, so these tests check
the harness rather than the agent: that every task starts genuinely broken, that
the correct fix makes it pass, and that the grader cannot be satisfied by an
agent which claims success without doing the work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cagent.cli.pricing import Price
from cagent.types import Usage
from eval.run import Attempt, Report, main, materialise, print_report, verify
from eval.tasks import TASKS, Task, task_by_id


class TestTaskDefinitions:
    def test_task_ids_are_unique(self) -> None:
        ids = [task.id for task in TASKS]
        assert len(ids) == len(set(ids))

    def test_every_task_has_a_prompt_and_a_check(self) -> None:
        for task in TASKS:
            assert task.prompt.strip(), task.id
            assert task.verify.strip(), task.id
            assert task.files or task.binary_files, task.id

    def test_no_task_reveals_its_verification_command(self) -> None:
        # Telling the agent the grader's command invites writing to the grader
        # rather than fixing the bug.
        for task in TASKS:
            assert task.verify not in task.prompt, task.id

    def test_lookup_by_id_works_and_reports_unknown_names(self) -> None:
        assert task_by_id("sign-error").id == "sign-error"
        with pytest.raises(KeyError) as caught:
            task_by_id("no-such-task")
        assert "sign-error" in str(caught.value)  # the message lists real ids


@pytest.mark.parametrize("task", TASKS, ids=[task.id for task in TASKS])
class TestTasksAreWellFormed:
    def test_the_task_starts_broken(self, task: Task, tmp_path: Path) -> None:
        # A task that passes before the agent touches it measures nothing.
        materialise(task, tmp_path)
        passed, output = verify(task, tmp_path)
        assert not passed, f"{task.id} passes its own check before any fix:\n{output}"

    def test_the_check_produces_diagnostic_output(self, task: Task, tmp_path: Path) -> None:
        # The failure text is what the agent reads to locate the bug.
        materialise(task, tmp_path)
        _, output = verify(task, tmp_path)
        assert output.strip(), f"{task.id} fails silently, giving the agent nothing to go on"

    def test_files_are_written_as_specified(self, task: Task, tmp_path: Path) -> None:
        materialise(task, tmp_path)
        for name in task.files:
            assert (tmp_path / name).is_file()
        for name, data in task.binary_files.items():
            assert (tmp_path / name).read_bytes() == data


# The intended fix for each task, applied mechanically. This is how the harness
# proves a task is solvable at all — if these do not pass, a failing agent run
# tells us nothing about the agent.
SOLUTIONS: dict[str, dict[str, str]] = {
    "sign-error": {"calc.py": "def add(a, b):\n    return a + b\n"},
    "wrong-function": {
        "stats.py": (
            "def mean(values):\n"
            "    return sum(values) / len(values)\n\n\n"
            "def total(values):\n"
            "    return sum(values)\n\n\n"
            "def largest(values):\n"
            "    return max(values)\n"
        )
    },
    "two-files": {
        "client.py": (
            "def retrieve(url):\n"
            '    """Pretend to fetch a URL."""\n'
            '    return f"contents of {url}"\n'
        ),
        "service.py": (
            "from client import retrieve\n\n\ndef load(url):\n    return retrieve(url).upper()\n"
        ),
    },
    "crash-traceback": {
        "report.py": (
            "def summarise(rows):\n"
            "    return f'{len(rows)} rows, first is {rows[0]}'\n\n\n"
            "if __name__ == '__main__':\n"
            "    print(summarise(['alpha', 'beta']))\n"
        )
    },
    "missing-feature": {
        "text_utils.py": (
            "def shout(text):\n"
            '    """Return text in upper case with an exclamation mark."""\n'
            "    return text.upper() + '!'\n\n\n"
            "def titlecase(text):\n"
            '    """Capitalise the first letter of every word."""\n'
            "    return ' '.join(word.capitalize() for word in text.split())\n"
        )
    },
    "ambiguous-string": {
        "config.py": (
            "class HttpSettings:\n"
            "    timeout = 5\n"
            "    retries = 3\n\n\n"
            "class CacheSettings:\n"
            "    timeout = 5\n"
            "    size = 100\n\n\n"
            "class DatabaseSettings:\n"
            "    timeout = 30\n"
            "    pool = 10\n\n\n"
            "class QueueSettings:\n"
            "    timeout = 5\n"
            "    depth = 20\n"
        )
    },
}


class TestTasksAreSolvable:
    @pytest.mark.parametrize(
        "task", [t for t in TASKS if t.id in SOLUTIONS], ids=list(SOLUTIONS)
    )
    def test_the_intended_fix_passes(self, task: Task, tmp_path: Path) -> None:
        materialise(task, tmp_path)
        for name, text in SOLUTIONS[task.id].items():
            (tmp_path / name).write_text(text, encoding="utf-8", newline="")
        passed, output = verify(task, tmp_path)
        assert passed, f"{task.id} is not solvable by its intended fix:\n{output}"

    def test_the_encoding_task_passes_only_with_the_bytes_preserved(
        self, tmp_path: Path
    ) -> None:
        # The point of the task: the correct edit keeps the BOM and the CRLFs.
        task = task_by_id("crlf-and-bom")
        materialise(task, tmp_path)
        target = tmp_path / "greet.py"

        careful = target.read_bytes().replace(b"'Hello'", b"'Welcome'")
        target.write_bytes(careful)
        passed, output = verify(task, tmp_path)
        assert passed, output

    def test_the_encoding_task_catches_a_careless_rewrite(self, tmp_path: Path) -> None:
        task = task_by_id("crlf-and-bom")
        materialise(task, tmp_path)
        target = tmp_path / "greet.py"

        # What a naive whole-file write produces: LF endings, no BOM.
        text = target.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
        target.write_bytes(text.replace("'Hello'", "'Welcome'").encode())

        passed, output = verify(task, tmp_path)
        assert not passed
        assert "BOM" in output or "CRLF" in output

    def test_the_precision_task_catches_a_replace_all(self, tmp_path: Path) -> None:
        # The failure mode the task exists to detect.
        task = task_by_id("ambiguous-string")
        materialise(task, tmp_path)
        target = tmp_path / "config.py"
        target.write_text(
            target.read_text(encoding="utf-8").replace("timeout = 5", "timeout = 30"),
            encoding="utf-8",
        )
        passed, _ = verify(task, tmp_path)
        assert not passed

    def test_the_localisation_task_catches_editing_the_wrong_function(
        self, tmp_path: Path
    ) -> None:
        task = task_by_id("wrong-function")
        materialise(task, tmp_path)
        target = tmp_path / "stats.py"
        # "Fixing" mean() instead of largest().
        target.write_text(
            target.read_text(encoding="utf-8").replace(
                "def mean(values):\n    return sum(values) / len(values)",
                "def mean(values):\n    return sum(values) / max(len(values), 1)",
            ),
            encoding="utf-8",
        )
        passed, _ = verify(task, tmp_path)
        assert not passed


class TestGrading:
    def test_a_claim_of_success_does_not_pass(self, tmp_path: Path) -> None:
        # The reason grading ignores the agent's prose entirely.
        task = task_by_id("sign-error")
        materialise(task, tmp_path)
        (tmp_path / "REPORT.md").write_text("I fixed the bug. All tests pass.\n", encoding="utf-8")
        passed, _ = verify(task, tmp_path)
        assert not passed

    def test_deleting_the_test_does_not_pass(self, tmp_path: Path) -> None:
        # pytest exits non-zero when it collects nothing, so the obvious cheat
        # is caught by the check itself.
        task = task_by_id("sign-error")
        materialise(task, tmp_path)
        (tmp_path / "test_calc.py").unlink()
        passed, _ = verify(task, tmp_path)
        assert not passed

    def test_a_verification_timeout_is_a_failure_not_a_hang(self, tmp_path: Path) -> None:
        slow = Task(
            id="slow",
            prompt="x",
            files={"loop.py": "while True:\n    pass\n"},
            verify="python loop.py",
        )
        materialise(slow, tmp_path)
        import eval.run as runner

        original = runner.VERIFY_TIMEOUT
        runner.VERIFY_TIMEOUT = 1.0
        try:
            passed, output = verify(slow, tmp_path)
        finally:
            runner.VERIFY_TIMEOUT = original
        assert not passed and "timed out" in output

    def test_a_missing_command_is_a_failure_not_a_crash(self, tmp_path: Path) -> None:
        broken = Task(
            id="broken", prompt="x", files={"a.py": "pass\n"}, verify="definitely-not-a-command"
        )
        materialise(broken, tmp_path)
        passed, _ = verify(broken, tmp_path)
        assert not passed

    def test_verification_preserves_unicode_output(self, tmp_path: Path) -> None:
        task = Task(
            id="unicode",
            prompt="x",
            files={"unicode_output.py": "print('中文路径')\n"},
            verify="python unicode_output.py",
        )
        materialise(task, tmp_path)
        passed, output = verify(task, tmp_path)
        assert passed and "中文路径" in output

    def test_verification_uses_the_current_interpreter(self, tmp_path: Path) -> None:
        task = Task(
            id="interpreter",
            prompt="x",
            files={
                "check_interpreter.py": (
                    "import pathlib, sys\n"
                    f"expected = {str(Path(sys.executable).parent)!r}\n"
                    "assert pathlib.Path(sys.executable).parent == pathlib.Path(expected)\n"
                )
            },
            verify="python check_interpreter.py",
        )
        materialise(task, tmp_path)
        passed, output = verify(task, tmp_path)
        assert passed, output

    def test_stale_bytecode_cannot_produce_a_false_failure(self, tmp_path: Path) -> None:
        # Same hazard the shell tool guards: a same-size fix inside one second
        # would otherwise be graded against the old bytecode.
        task = task_by_id("sign-error")
        materialise(task, tmp_path)
        assert not verify(task, tmp_path)[0]
        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        passed, output = verify(task, tmp_path)
        assert passed, output


class TestReporting:
    def test_the_pass_rate_is_computed(self) -> None:
        report = Report(model="deepseek-chat")
        for passed in (True, True, False, True):
            report.attempts.append(
                Attempt(
                    task_id="t",
                    passed=passed,
                    steps=3,
                    tools_used=["read_file"],
                    usage=Usage(100, 20),
                    elapsed_s=1.0,
                    verify_output="",
                    reply="",
                )
            )
        assert report.passed == 3
        assert report.pass_rate == 0.75
        assert report.total_usage.prompt_tokens == 400

    def test_attempts_are_grouped_per_task(self) -> None:
        report = Report()
        for task_id in ("a", "b", "a"):
            report.attempts.append(
                Attempt(
                    task_id=task_id,
                    passed=True,
                    steps=1,
                    tools_used=[],
                    usage=Usage(),
                    elapsed_s=0.0,
                    verify_output="",
                    reply="",
                )
            )
        grouped = report.per_task()
        assert len(grouped["a"]) == 2 and len(grouped["b"]) == 1

    def test_the_printed_report_shows_results_and_totals(self, capsys) -> None:
        report = Report(
            model="some-model",
            prices={"some-model": Price(1.0, 4.0)},
        )
        report.attempts.append(
            Attempt(
                task_id="sign-error",
                passed=True,
                steps=4,
                tools_used=["read_file", "edit_file", "run_bash"],
                usage=Usage(5000, 300),
                elapsed_s=12.3,
                verify_output="2 passed",
                reply="Fixed it.",
            )
        )
        report.attempts.append(
            Attempt(
                task_id="two-files",
                passed=False,
                steps=20,
                tools_used=["read_file"],
                usage=Usage(9000, 500),
                elapsed_s=30.0,
                verify_output="ImportError: cannot import name 'retrieve'",
                reply="I could not finish.",
                error="MaxStepsExceeded",
            )
        )
        print_report(report)
        out = capsys.readouterr().out
        assert "pass@1: 1/2 (50%)" in out
        assert "sign-error" in out and "two-files" in out
        assert "MaxStepsExceeded" in out
        assert "retrieve" in out  # the failing check's last line
        assert "$" in out  # a rate was configured
        assert "some-model" in out

    def test_an_attempt_serialises_for_the_json_report(self) -> None:
        attempt = Attempt(
            task_id="t",
            passed=True,
            steps=2,
            tools_used=["read_file"],
            usage=Usage(10, 5),
            elapsed_s=1.5,
            verify_output="ok",
            reply="done",
        )
        record = attempt.as_record()
        assert record["task"] == "t" and record["passed"] is True
        assert record["prompt_tokens"] == 10


class TestRunnerCli:
    def test_list_prints_every_task(self, capsys) -> None:
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        for task in TASKS:
            assert task.id in out

    def test_an_unknown_task_is_rejected(self, capsys) -> None:
        assert main(["--task", "not-a-task"]) == 2
        assert "not-a-task" in capsys.readouterr().err

    def test_an_incomplete_endpoint_explains_itself(self, capsys) -> None:
        # ``isolate_config`` points home and the working directory at an empty
        # directory, so nothing is configured and nothing can be inherited.
        assert main(["--task", "sign-error"]) == 2
        err = capsys.readouterr().err
        assert "base_url" in err and "model" in err
        assert ".cagent.toml" in err

    def test_the_module_is_runnable(self) -> None:
        # `python -m eval.run --list` is the documented entry point.
        completed = subprocess.run(
            [sys.executable, "-m", "eval.run", "--list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": "src"},
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert "sign-error" in completed.stdout
