"""Benchmark tasks: what the agent is asked to fix, and how it is graded.

Each task is a small broken project plus a verification command. Grading is
mechanical — the agent's own claims are ignored entirely, and only the exit code
of a check it did not write decides pass or fail. That is the whole design: an
agent that says "fixed it" is not evidence, and a benchmark that trusts prose
measures nothing.

The tasks are deliberately small but not toys. Each one reproduces a specific
failure that a real coding agent has to survive:

* a bug where the fix is one operator, but only after reading the test;
* a bug in one of several similar-looking functions, so the wrong edit passes
  nothing;
* a change that has to happen in two files at once;
* a bug whose only evidence is a traceback from a crash;
* a missing feature specified entirely by a failing test;
* an ambiguous string that appears many times, so a naive search-and-replace
  edits the wrong line;
* a file with CRLF endings and a BOM, where a careless write corrupts every line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TASKS", "Task", "task_by_id"]


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark case."""

    id: str
    prompt: str
    """What the user asks. Deliberately as vague as a real request would be:
    naming the file and the line would be doing the agent's work for it."""

    files: dict[str, str]
    """Initial workspace contents, path -> text."""

    verify: str
    """Shell command whose zero exit code means success. The agent is never
    told this command, so it cannot special-case it."""

    tags: tuple[str, ...] = ()
    binary_files: dict[str, bytes] = field(default_factory=dict)
    """Files written byte-for-byte, for encoding and line-ending cases."""



TASKS: tuple[Task, ...] = (
    Task(
        id="sign-error",
        prompt="The tests in this project are failing. Find out why and fix it.",
        tags=("read-test", "one-line-fix"),
        files={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": (
                "from calc import add\n\n\n"
                "def test_positive():\n    assert add(2, 3) == 5\n\n\n"
                "def test_zero():\n    assert add(7, 0) == 7\n"
            ),
        },
        verify="python -m pytest -q",
    ),
    Task(
        id="wrong-function",
        prompt="Run the test suite and fix whatever is broken.",
        tags=("localisation",),
        files={
            # Three near-identical functions: editing the wrong one fixes nothing,
            # so the agent has to read the failure rather than pattern-match.
            "stats.py": (
                "def mean(values):\n"
                "    return sum(values) / len(values)\n\n\n"
                "def total(values):\n"
                "    return sum(values)\n\n\n"
                "def largest(values):\n"
                "    return min(values)\n"
            ),
            "test_stats.py": (
                "from stats import largest, mean, total\n\n\n"
                "def test_mean():\n    assert mean([2, 4]) == 3\n\n\n"
                "def test_total():\n    assert total([1, 2, 3]) == 6\n\n\n"
                "def test_largest():\n    assert largest([3, 9, 4]) == 9\n"
            ),
        },
        verify="python -m pytest -q",
    ),
    Task(
        id="two-files",
        prompt=(
            "Rename the `fetch` function to `retrieve` everywhere it is used, "
            "keeping the tests passing."
        ),
        tags=("multi-file", "search"),
        files={
            "client.py": (
                "def fetch(url):\n"
                '    """Pretend to fetch a URL."""\n'
                '    return f"contents of {url}"\n'
            ),
            "service.py": (
                "from client import fetch\n\n\n"
                "def load(url):\n"
                "    return fetch(url).upper()\n"
            ),
            "test_service.py": (
                "from client import retrieve\nfrom service import load\n\n\n"
                "def test_retrieve_exists():\n"
                '    assert retrieve("x") == "contents of x"\n\n\n'
                "def test_load_uses_it():\n"
                '    assert load("x") == "CONTENTS OF X"\n'
            ),
        },
        verify="python -m pytest -q",
    ),
    Task(
        id="crash-traceback",
        prompt="Running the report script raises an exception. Make it run cleanly.",
        tags=("traceback", "run-to-reproduce"),
        files={
            # The bug is a real AttributeError, not a silently wrong value: the
            # only evidence is the traceback, and nothing here mentions `len`.
            # `check_report.py` is what grades it, and the prompt does not name
            # the command, so the agent cannot write to the grader.
            "report.py": (
                "def summarise(rows):\n"
                "    return f'{rows.length} rows, first is {rows[0]}'\n\n\n"
                "if __name__ == '__main__':\n"
                "    print(summarise(['alpha', 'beta']))\n"
            ),
            "check_report.py": (
                "import subprocess\n"
                "import sys\n\n"
                "done = subprocess.run(\n"
                "    [sys.executable, 'report.py'], capture_output=True, text=True\n"
                ")\n"
                "assert done.returncode == 0, done.stderr\n"
                "out = done.stdout.strip()\n"
                "assert out == '2 rows, first is alpha', out\n"
                "print('report ok')\n"
            ),
        },
        verify="python check_report.py",
    ),
    Task(
        id="missing-feature",
        prompt="Make the test suite pass.",
        tags=("write-new-code",),
        files={
            "text_utils.py": (
                "def shout(text):\n"
                '    """Return text in upper case with an exclamation mark."""\n'
                "    return text.upper() + '!'\n"
            ),
            "test_text_utils.py": (
                "from text_utils import shout, titlecase\n\n\n"
                "def test_shout():\n    assert shout('hi') == 'HI!'\n\n\n"
                "def test_titlecase():\n"
                "    assert titlecase('hello wide world') == 'Hello Wide World'\n\n\n"
                "def test_titlecase_handles_empty():\n"
                "    assert titlecase('') == ''\n"
            ),
        },
        verify="python -m pytest -q",
    ),
    Task(
        id="ambiguous-string",
        prompt=(
            "In config.py, the default timeout for the database connection should "
            "be 30 instead of 5. Change only that one."
        ),
        tags=("precision", "ambiguity"),
        files={
            # "timeout = 5" appears four times: a replace-all destroys the file,
            # and an unanchored search-and-replace edits the wrong line.
            "config.py": (
                "class HttpSettings:\n"
                "    timeout = 5\n"
                "    retries = 3\n\n\n"
                "class CacheSettings:\n"
                "    timeout = 5\n"
                "    size = 100\n\n\n"
                "class DatabaseSettings:\n"
                "    timeout = 5\n"
                "    pool = 10\n\n\n"
                "class QueueSettings:\n"
                "    timeout = 5\n"
                "    depth = 20\n"
            ),
            "test_config.py": (
                "import config\n\n\n"
                "def test_database_changed():\n"
                "    assert config.DatabaseSettings.timeout == 30\n\n\n"
                "def test_others_untouched():\n"
                "    assert config.HttpSettings.timeout == 5\n"
                "    assert config.CacheSettings.timeout == 5\n"
                "    assert config.QueueSettings.timeout == 5\n"
                "    assert config.HttpSettings.retries == 3\n"
                "    assert config.DatabaseSettings.pool == 10\n"
            ),
        },
        verify="python -m pytest -q",
    ),
    Task(
        id="crlf-and-bom",
        prompt="In greet.py, change the greeting from 'Hello' to 'Welcome'.",
        tags=("encoding", "line-endings"),
        files={},
        binary_files={
            # UTF-8 BOM, CRLF endings, and a non-ASCII comment. A careless
            # whole-file rewrite converts every line and the check fails.
            "greet.py": (
                "﻿# 问候语模块\r\n"
                "GREETING = 'Hello'\r\n"
                "\r\n"
                "\r\n"
                "def greet(name):\r\n"
                "    return f'{GREETING}, {name}!'\r\n"
            ).encode(),
            "check_encoding.py": (
                "import sys\n"
                "raw = open('greet.py', 'rb').read()\n"
                "assert raw.startswith(b'\\xef\\xbb\\xbf'), 'the BOM was stripped'\n"
                "assert b'\\r\\n' in raw, 'CRLF line endings were converted to LF'\n"
                "assert raw.count(b'\\n') == raw.count(b'\\r\\n'), 'line endings are now mixed'\n"
                "text = raw.decode('utf-8-sig')\n"
                "assert 'Welcome' in text, 'the greeting was not changed'\n"
                "assert 'Hello' not in text, 'the old greeting is still there'\n"
                "assert '问候语模块' in text, 'the non-ASCII comment was mangled'\n"
                "sys.path.insert(0, '.')\n"
                "import greet\n"
                "assert greet.greet('Ada') == 'Welcome, Ada!', greet.greet('Ada')\n"
                "print('encoding preserved')\n"
            ).encode(),
        },
        verify="python check_encoding.py",
    ),
)


def task_by_id(task_id: str) -> Task:
    """Look up one task.

    Raises:
        KeyError: If no task has that id.
    """
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(f"no benchmark task named {task_id!r}; have: {[t.id for t in TASKS]}")
