# cagent

A coding agent, written from scratch. It talks to a language model, reads and
edits files, runs commands, reads what they printed, and keeps going until the
task is done or a limit stops it.

No agent framework or SDK is used. The model is reached over raw HTTP; the tool
schemas, the streaming parser, the conversation history, the context budget, the
loop, and the stopping conditions are all hand-written and are the substance of
the project.

```
$ cagent "the tests are failing, find out why and fix it"

· step 1/40 · 1,757 tokens (1% of window)
⏺ run_bash(python -m pytest -q, description=see the failure)
  ⎿ ✗ exit 1 (0.68s)
    E       assert -1 == 5
    E        +  where -1 = add(2, 3)
· step 2/40 · 1,914 tokens (1% of window)
⏺ read_file(calc.py)
  ⎿ ✓ 2 of 2 lines (0.00s)
· step 3/40 · 1,959 tokens (2% of window)
⏺ edit_file(calc.py, old_string=    return a - b, new_string=    return a + b)
  ⎿ ✓ +1/-1 (0.00s)
    --- a/calc.py
    +++ b/calc.py
    @@ -1,2 +1,2 @@
     def add(a, b):
    -    return a - b
    +    return a + b
· step 4/40 · 2,036 tokens (2% of window)
⏺ run_bash(python -m pytest -q, description=confirm green)
  ⎿ ✓ exit 0 (0.63s)
    2 passed in 0.00s
· step 5/40 · 2,097 tokens (2% of window)

add subtracted instead of adding (calc.py:2). Fixed and pytest passes.

┌────────────── session finished ───────────────┐
│ steps                                       5 │
│ tools run                                   4 │
│ elapsed                                  1.5s │
│ prompt tokens                           4,500 │
│ completion tokens                         225 │
│ cost               no rate set for this model │
│ files touched                         calc.py │
└───────────────────────────────────────────────┘
```

---

## Running it

Requires Python 3.11+.

```bash
pip install -e .
```

### Configuring an endpoint

Three settings, and none of them are guessed. Copy `.cagent.example.toml` to
`.cagent.toml` (gitignored) and fill in:

```toml
[cagent]
base_url = "https://api.example.com/v1"        # where to send requests
model = "the-model-your-endpoint-serves"       # what to ask for
api_key = "..."                                # your key
```

Anything OpenAI-compatible works — a vendor API, a gateway, a proxy, or a local
server. Every key in that table is a field name, and flags of the same name
(`--base-url`, `--model`) override it for one run: the layers are `~/.cagent.toml`
→ `./.cagent.toml` → flags, later winning. There is no environment layer, on
purpose — `CAGENT_MAX_STEPS` beside `max_steps` is one setting with two
spellings, which is one place too many to look when the agent does something
unexpected.

Details worth knowing:

- **`base_url` is used verbatim.** `/chat/completions` is appended (or
  `/messages` on the Anthropic wire), so include whatever version segment your
  provider publishes — usually `.../v1`. Nothing is inserted or guessed, so a
  gateway that mounts its API somewhere unusual still works; omitting a needed
  `/v1` is the usual cause of a 404 on the first request. A trailing slash is
  handled either way.
- **`--wire` picks the request format**, `openai` (the default, which almost
  everything emulates) or `anthropic` for the Messages API. This is the only
  seam where a genuinely different protocol has to be declared; everything else
  is just a URL.
- **`--no-key` for a local server.** Ollama or llama.cpp then gets no
  `Authorization` header at all, rather than one containing the word `None`.
- **Unknown keys are an error.** A misspelled setting fails loudly at startup
  instead of being silently ignored, which is the failure mode that costs an
  afternoon.

The key comes from a config file and nowhere else: never from a flag, because a
flag lands in shell history and the process list. Keep `.cagent.toml` untracked
(it is gitignored here), or put the key in `~/.cagent.toml` and leave the project
file for everything else. `cagent --show-config` prints what resolved —
endpoint, model, wire, the full request URL, and whether a key was found — with
the key masked. It also works when the configuration is incomplete, which is
when you need it.

### Using it

```bash
cagent "add a --json flag to the report command"     # one task
cagent                                               # interactive session
cagent --list-tools                                  # tools and their arguments
```

To continue a previous conversation, start an interactive session with `cagent`,
then enter `/resume TRACE`. Use `/help resume` inside the session for details.
The trace is read-only; the current configuration, credentials, workspace,
approvals, and sandbox remain active. A trace cannot restore an old Docker
container or unsynchronised sandbox files, and clipped tool output means
recovery is best-effort rather than a filesystem snapshot.

The command is installed by the Python package; it is not tied to this
repository. On Windows, activate the virtual environment once per terminal and
then run it from any project directory:

```powershell
& "C:\path\to\CodingAgent\venv\Scripts\Activate.ps1"
cd D:\Projects\MyProject
cagent "run the tests and fix the failure"
```

To avoid activating a virtual environment manually, install the project with
`pipx install --editable C:\path\to\CodingAgent`; `pipx` exposes `cagent` on
your user `PATH`. You can also keep the shell in another directory and select
the target explicitly:

```powershell
cagent --workspace D:\Projects\MyProject "inspect and fix the tests"
```

#### Interactive commands

Inside the interactive session, `/help` shows a compact command list. Add a
command name to see its details, for example `/help sandbox` or `/help resume`.
Conversation recovery is available only inside the session:

```text
/resume .cagent/traces/<id>.jsonl
```

This replaces the current conversation history with the messages recorded in
the trace and keeps using the current workspace and configuration. It does not
restore the old container, unsynchronised sandbox files, or approval state.

#### 交互式沙箱

不需要重启程序即可在交互窗口中开启、提交或撤销沙箱修改。输入
`/help sandbox` 可以在程序内查看沙箱操作说明，`/help resume` 可以查看对话恢复说明。

| 命令 | 作用 |
| --- | --- |
| `/sandbox` | 查看当前状态、镜像和同步策略 |
| `/sandbox on [IMAGE]` | 创建项目快照并开启 Docker 隔离；不写 IMAGE 时使用当前配置 |
| `/sandbox image IMAGE` | 设置本地镜像；沙箱开启时会重启容器，但保留当前快照 |
| `/sandbox sync never\|ask\|always` | 设置关闭沙箱或退出程序时的处理方式 |
| `/sandbox apply` | 立即把当前修改同步到真实项目，沙箱继续运行，并建立新基线 |
| `/sandbox rollback` | 丢弃尚未同步的修改，重新载入真实项目，沙箱继续运行 |
| `/sandbox off` | 按当前同步策略处理修改，然后回到真实项目模式 |

推荐的交互流程：

```text
/sandbox image my-project-agent:latest
/sandbox sync ask
/sandbox on
... 让 Agent 修改代码并运行测试 ...
/sandbox apply       # 阶段性提交，继续在沙箱中工作
... 继续修改 ...
/sandbox rollback    # 放弃最近一轮尚未提交的修改
/sandbox off         # 完成后回到真实项目
```

同步策略的含义：

- `never`：关闭或退出时丢弃所有未提交的沙箱修改。
- `ask`：关闭或退出时展示受限 diff，确认后才同步。
- `always`：关闭或退出时自动同步；若真实项目被其他程序改动，会拒绝同步并提示冲突。
- `/sandbox apply` 是显式同步命令，不再弹出第二次确认；`/sandbox rollback` 不会撤销已经 apply 的修改。

沙箱生命周期属于“一个 Agent 对话会话”：同一个交互窗口中的多个任务共享
一个快照和一个容器，容器在第一次 `run_bash` 时才启动，退出窗口后删除。相同
工作目录同时打开两个 Agent，会分别拥有自己的快照和容器，未同步的修改互不
可见。沙箱开启期间，文件工具操作临时快照，Shell 在 Linux 容器中运行；真实
项目只有在同步时才会被修改，默认的工作目录边界和命令审批仍然有效。

With `--workspace`, that directory is both the file/command sandbox and the
location where the project `.cagent.toml` is read. The user-level
`%USERPROFILE%\.cagent.toml` remains available for shared endpoint settings.

**Supervision.** `--approval suggest` confirms every change; `auto-edit` (the
default) lets file edits through and confirms shell commands; `full-auto`
confirms only destructive commands. Nothing auto-approves a destructive command.

**Sandboxing.** Shell commands normally run in the project directory. For an
isolated run, use `--sandbox docker --sandbox-sync ask` (or set the same fields
in `.cagent.toml`). The agent copies the project to a temporary snapshot and
mounts only that snapshot into a constrained, network-disabled Docker
container. The real project is never mounted into the container. One Agent
session owns one container: the first `run_bash` starts it and later commands
use `docker exec`, so tools and packages installed during that session remain
available in writable locations. The root filesystem is read-only, so stable
project dependencies should be baked into the image rather than installed by
the Agent. The container is removed when the Agent closes; opening a new Agent
creates a new container and a fresh snapshot. Docker reuses local image layers,
so it does not reinstall image dependencies on every session.

The image must already exist locally because pulls are disabled. For a project
with non-Python or heavier dependencies, create a project-specific image once:

```dockerfile
# Dockerfile.agent
FROM node:22-bookworm
RUN npm install -g pnpm
```

```powershell
docker build -f Dockerfile.agent -t my-project-agent:latest .
cagent --workspace . --sandbox docker --sandbox-image my-project-agent:latest "run the tests"
```

Keep credentials out of the Docker build context. At minimum, add
`.cagent.toml`, `.cagent.toml.*`, `.git`, and `.cagent/` to `.dockerignore`;
the example Dockerfile above does not need to copy the project into the image.

The image build is an explicit user action because a Dockerfile can execute
arbitrary installation scripts. At session end, `never` discards the snapshot,
`ask` shows a bounded diff and requires a separate approval, and `always` copies
it back after a concurrent-change check. Docker Desktop/Engine must be running;
if it is unavailable, the sandbox command fails closed rather than falling back
to the host.

**Cost.** Token counts are always reported. Dollar figures require rates you
supply, because a built-in price table goes stale and a stale price prints a
confident number that is wrong:

```toml
[cagent.prices."the-model-your-endpoint-serves"]
input_per_m = 0.27          # US dollars per million tokens
output_per_m = 1.10
cached_input_per_m = 0.07   # optional
```

### Checking it

```bash
pytest                       # 533 tests
ruff check src tests eval
mypy src/cagent eval
python -m eval.run           # the benchmark; needs a real endpoint
```

---

## Architecture

Five layers. Each one depends only on the layer below it, and the seams are where
the design decisions live.

```
┌──────────────────────────────────────────────────────────────────┐
│  cli/          argument parsing · streaming render · approval    │
│                prompt · cost                                     │
│                the only layer that prints                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ typed events
┌────────────────────────────▼─────────────────────────────────────┐
│  agent/        the loop · context compaction · loop guards ·     │
│                approval policy · repo map · system prompt ·      │
│                JSONL trace                                      │
└──────────────┬──────────────────────────────────┬────────────────┘
               │ Message / ToolSpec               │ ToolOutcome
┌──────────────▼──────────────┐   ┌───────────────▼────────────────┐
│  llm/                       │   │  tools/                        │
│  provider ABC · OpenAI wire │   │  BaseTool · schema reflection · │
│  Anthropic wire · SSE       │   │  registry · match ladder ·     │
│  parser · tool-call         │   │  diffs · files · shell ·       │
│  accumulator · retry ·      │   │  search · truncation           │
│  token estimation           │   │                                │
└──────────────┬──────────────┘   └───────────────┬────────────────┘
               └──────────────┬───────────────────┘
┌─────────────────────────────▼────────────────────────────────────┐
│  types · errors · config     provider-neutral message model,     │
│                              exception hierarchy, layered config │
└──────────────────────────────────────────────────────────────────┘
```

### The loop

One step: ask the model what to do. If it answers in prose, the turn is over. If
it asks for tools, authorise each call, run it, append the result, and ask again
with what was learned.

Two invariants make that survivable.

**Every tool call gets exactly one result.** Including refused calls, calls with
malformed JSON, and calls to tools that do not exist. Both wire formats reject a
request whose `tool_calls` have no matching results, so an unanswered call does
not degrade the run — it ends it with a 400. Every early exit in dispatch still
produces a result part.

**Nothing a tool does can stop the loop.** `BaseTool.invoke` converts bad
arguments, a refused path, and unexpected exceptions into a `ToolOutcome` with
`is_error` set. The engine does the same for approval refusals and its own
dispatch errors. A failure is an observation the model can act on.

Those two combine into the property the agent is actually built around: a failing
command comes back as text with its traceback intact, and the model fixes its own
mistake. Self-correction is not a feature added on top; it is what happens when
failures are reported instead of raised.

### The edit engine — `tools/matching.py`

An agent that rewrites whole files loses code it never read. So edits are local:
`edit_file(path, old_string, new_string)`.

The problem is that `old_string` is the model's *copy* of what it read, and copies
drift — indentation gets normalised, tabs become spaces, a character is mistyped.
A byte-exact matcher fails on all of it. So matching degrades through three
levels, returning every match from the first level that finds any, never mixing
levels:

| level | tolerates | similarity |
|---|---|---|
| exact | nothing | 1.00 |
| whitespace | uniform indent drift, tabs vs spaces, trailing whitespace | 0.99 |
| fuzzy | small typos, one inserted or deleted line | ≥ threshold (0.86) |

Two details make it usable rather than dangerous:

- **Re-indentation.** When a match was found at a different indentation than the
  needle, the replacement is shifted by the same delta. Without this a fuzzy
  replace lands mis-indented and the file no longer parses — which is why naive
  fuzzy replace is worse than no fuzzy replace.
- **Ambiguity is refused, not guessed.** Two matches means an error listing both
  line numbers and telling the model to include more context. `replace_all` is
  honoured only for exact matches; on a fuzzy match it is refused outright,
  because a fuzzy match applied everywhere is how an agent destroys a file.

A failed match is not just "not found": `best_rejected` reports the nearest
below-threshold candidate with its line number and score, so the model can see
what it got wrong.

Edits are atomic (temp file + `os.replace`) and preserve the file's newline style
and encoding. An edit that silently converts a CRLF file to LF shows up in the
user's version control as a whole-file change.

### Context management — `agent/context.py`

Each step adds an assistant turn and its tool results, so a long task grows its
own prompt until the request is rejected. Two constraints shape the fix.

*Tool calls and their results are inseparable* — so history is segmented into
**steps** (an assistant turn plus the results answering it) and steps are the
unit of removal. Segmenting per *user turn* instead is the obvious mistake: an
agentic task is one user message followed by dozens of steps, so it produces a
single indivisible block and compaction can never free anything during exactly
the long task that needs it.

*Not all history is equally valuable* — so compaction escalates, cheapest first:

1. **elide** old tool output, keeping a few lines and a marker. Usually enough,
   because tool output is the bulk of a transcript and stale output is its least
   useful part.
2. **summarise** old steps into a progress note, via a separate tool-free model
   call. Rejected if the note would not be smaller than what it replaces.
3. **drop** old steps, leaving a marker saying how many went and instructing the
   model to re-read rather than trust its memory.

The first block — the task — is never touched, and the last few steps are kept
verbatim. An agent that forgets what it was asked will confidently finish the
wrong job.

### Stopping — `agent/guards.py`

An agent that decides its own next action can fail by never stopping, and that
failure is expensive rather than loud. Three independent bounds: a step ceiling,
a token budget, and repetition detection.

Repetition is handled in two stages. A repeated identical call first earns a
*nudge* — a tool result telling the model it is looping and what to try instead —
because the usual cause is a model that cannot see its own pattern, and one
sentence often fixes it. Only an unbroken streak past the limit stops the run.

### Safety — `agent/approval.py`, `tools/shell.py`

`classify_command` sorts a command into safe / mutating / dangerous by inspecting
each stage of a compound command and taking the worst; the allowlist is small, so
it can only be wrong in the safe direction. The approval policy holds all the
permission rules in one auditable place.

The approval prompt shows the tool's own dry-run account of the action — for an
edit, the *real* diff of what would be written. A prompt the user cannot evaluate
trains them to approve everything, which is worse than no prompt.

A dangerous command's remembered-approval key is its full text, so consent never
transfers from one irreversible action to a different one. "Always allow" is
neither offered nor honoured for those.

Other measures: paths are resolved and checked for workspace containment (`..`
and symlinks both caught); command timeouts kill the whole process tree, not just
the shell; and the child environment is stripped of anything whose name looks
like a credential — the model can run `env`, and whatever it prints lands in the
transcript and the trace file.

### Schema reflection — `tools/schema.py`

A tool declares its arguments once, as an annotated dataclass, and the JSON Schema
sent to the model is derived from it. They cannot drift apart.

```python
@dataclass
class EditFileParams:
    path: Annotated[str, Doc("Path to the file")]
    old_string: Annotated[str, Doc("Text to find — must match uniquely")]
    new_string: Annotated[str, Doc("Replacement text")]
    replace_all: bool = False
```

Reflection covers primitives, `Literal`, `Enum`, `list[T]`, `dict[str, T]`,
`X | None`, and nested dataclasses (which `multi_edit` needs). An unsupported
annotation raises at schema-build time — failing loudly beats shipping a wrong
schema and debugging the resulting nonsense arguments.

Validation of the model's arguments is deliberately asymmetric: lenient where
models are reliably sloppy (`"12"` for an int, `"yes"` for a bool, a bare scalar
where an array was declared), strict where being wrong would run the wrong action
(unknown keys, missing required fields, values outside a `Literal`). All field
errors are reported together, and each message is phrased as an instruction the
model can act on — that string goes straight back into the transcript.

### Observability — `agent/trace.py`, `cli/render.py`

The engine never prints. It emits typed events to a sink, and the CLI decides what
a human should see; the same run drives a terminal renderer, a JSONL trace, and a
test that asserts on a list, concurrently.

The trace is one JSON object per event, flushed per line, because the runs worth
examining are the ones that crashed. `/resume TRACE` reconstructs the recorded
user, assistant, and tool-result messages in the current interactive Agent;
credentials, approvals, usage counters, and Docker state are deliberately not
stored in the trace.

The session summary reports tokens, cached tokens, and estimated cost; for a model
with no published rate it reports the token counts and no dollar figure rather
than guessing.

---

## Testing

482 tests, hermetic: no network, no real sleeps, no dependence on the machine's
PATH. `mypy` clean, `ruff` clean, 86% line coverage.

The interesting choices are in what gets tested. A scripted provider records the
exact request bodies the engine built, so the tests can assert on serialisation
and on the call/result pairing invariant across a whole run. The search tool is
tested through *both* engines with the same fixtures, asserting byte-identical
output, so behaviour does not depend on whether ripgrep is installed. And the bulk
of the suite is on the paths a live run produces only by accident: a refused edit,
malformed tool-call JSON, an unknown tool, a crashing tool, a looping model, a
full context window.

Two bugs the suite caught, both worth stating because neither is obvious:

- **Stale bytecode.** Python decides a cached `.pyc` is current from the source's
  mtime *in whole seconds* plus its size. Changing `a - b` to `a + b` alters
  neither — so a verification re-run inside the same second executed the old
  bytecode and reported the bug as unfixed, which would send the agent off to
  "fix" already-correct code. `run_bash` now sets `PYTHONDONTWRITEBYTECODE`.
- **A hanging timeout.** `subprocess.run(timeout=...)` kills only the shell; the
  surviving grandchild holds the output pipes open, so the wait never returns.
  The benchmark hung instead of scoring a loss. Both the shell tool and the
  grader now kill the whole process tree.

---

## Benchmark

```bash
python -m eval.run                 # all tasks, prints pass@1
python -m eval.run --repeat 3      # variance across attempts
python -m eval.run --json out.json # machine-readable results
```

Seven tasks, each a small broken project plus a verification command the agent is
never shown. Grading is mechanical: only the check's exit code counts, and the
agent's own report is recorded for reading but never consulted. An agent that says
"fixed it" is not evidence.

Each task targets a specific hazard rather than a generic bug:

| task | what it tests |
|---|---|
| `sign-error` | reading a failing test to locate a one-line fix |
| `wrong-function` | three near-identical functions — the wrong edit fixes nothing |
| `two-files` | a rename that must land in two files at once |
| `crash-traceback` | a bug whose only evidence is a traceback |
| `missing-feature` | code specified entirely by a failing test |
| `ambiguous-string` | a value appearing four times, where replace-all destroys the file |
| `crlf-and-bom` | CRLF + BOM + non-ASCII, where a careless rewrite corrupts every line |

`tests/test_eval.py` tests the *harness*: that every task starts genuinely broken,
that the intended fix passes, and that the obvious cheats — claiming success,
deleting the test, replace-all, a naive whole-file rewrite — are all caught. A
benchmark that silently mismeasures is worse than none.

---

## Trade-offs

**Token counting is estimated, not exact.** `tiktoken` is used when installed;
otherwise a hand-written heuristic that counts CJK codepoints separately, since a
per-character rule tuned on English badly underestimates Chinese. Estimates decide
when to compact; the numbers reported to the user come from the provider.

**Mid-stream failures are not retried.** Retries cover connecting and non-2xx
responses. Once tokens have arrived, a transport error surfaces to the engine —
replaying half a completion would corrupt the transcript.

**Command classification is a heuristic.** It decides what is worth interrupting
the user for, not what is permitted; the approval mode is the real gate. The
optional Docker mode adds a separate execution boundary, but it is still not a
kernel-grade security guarantee: Docker itself must be trusted and kept patched,
and the host project is protected by never mounting it read/write into the
container and by requiring a reviewed copy-back.

**The repo map is signatures only, and can go stale.** It is an index that tells
the model a symbol exists so it can search for it — never a substitute for reading
the file, which the system prompt says explicitly. It is rebuilt after the agent
writes, because a map that still describes the old tree is worse than none.

**Parallel tool calls execute sequentially.** The model may request several in one
step and all of them run, but in order. Concurrency would help on independent
reads and would need care around the approval prompt and shared file state.

**Fuzzy matching has a floor and no AST validation.** Below the similarity
threshold an edit is refused rather than guessed. Validating the result against a
parser — Tree-sitter, or `ast` for Python — would catch edits that match textually
but produce invalid syntax. Today the check is empirical: run the tests.
