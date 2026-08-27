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

┌───── session finished ─────┐
│ steps                    5 │
│ tools run                4 │
│ elapsed               1.6s │
│ prompt tokens        4,500 │
│ completion tokens      225 │
│ estimated cost     $0.0015 │
│ files touched      calc.py │
└────────────────────────────┘
```

---

## Running it

Requires Python 3.11+.

```bash
pip install -e .                      # or: pip install httpx rich
export DEEPSEEK_API_KEY=...           # or OPENAI_API_KEY, ANTHROPIC_API_KEY, …

cagent "add a --json flag to the report command"     # one task
cagent                                               # interactive session
cagent --show-config                                 # what got resolved, key masked
cagent --list-tools                                  # tools and their arguments
cagent --replay .cagent/traces/<id>.jsonl            # re-narrate a past run
```

The key is read from the environment or an untracked `.cagent.toml`, never from a
flag — a flag would put it in shell history and the process list.

### Pointing it at any endpoint

A preset is only a convenient default for three things: a base URL, a model, and
a request format. Anything OpenAI-compatible works without one — supply the
endpoint and the model directly:

```bash
cagent --base-url https://your-gateway.example.com/v1 \
       --model whatever-that-gateway-serves \
       "add a --json flag to the report command"
```

Equivalently, via the environment or `.cagent.toml`:

```bash
export CAGENT_BASE_URL=https://your-gateway.example.com/v1
export CAGENT_MODEL=whatever-that-gateway-serves
export CAGENT_API_KEY=...
```

```toml
[cagent]
base_url = "https://your-gateway.example.com/v1"
model = "whatever-that-gateway-serves"
```

Three details worth knowing:

- **`base_url` is used verbatim.** `/chat/completions` is appended (or
  `/messages` on the Anthropic wire), so include whatever version segment your
  provider publishes — usually `.../v1`. Nothing is guessed or inserted, because
  a gateway that mounts its API somewhere unusual should still work. Omitting a
  needed `/v1` is the usual cause of a 404 on the first request. A trailing slash
  is handled either way.
- **`--wire` picks the request format**, `openai` (the default) or `anthropic`.
  Set it only if the endpoint speaks the Anthropic Messages API. This is the one
  seam where a genuinely different protocol needs declaring; everything else is
  just a URL.
- **A local model needs no key.** `--provider ollama` sends no `Authorization`
  header at all, rather than a header containing the word `None`.

`cagent --show-config` prints exactly what got resolved — endpoint, model, wire,
and whether a key was found — with the key itself masked. It is the fastest way
to tell a configuration problem from a network one.

**Presets.** `--provider` supplies those defaults for `deepseek`, `openai`,
`anthropic`, `moonshot`, `dashscope`, `openrouter`, and `ollama`. A preset and an
explicit `--base-url` compose: the flag overrides just the endpoint and leaves
the preset's model and wire in place.

**Supervision.** `--approval suggest` confirms every change; `auto-edit` (the
default) lets file edits through and confirms shell commands; `full-auto`
confirms only destructive commands. Nothing auto-approves a destructive command.

```bash
pytest                       # 482 tests
ruff check src tests eval
mypy src/cagent eval
python -m eval.run           # the benchmark; needs a real key
```

---

## Architecture

Five layers. Each one depends only on the layer below it, and the seams are where
the design decisions live.

```
┌──────────────────────────────────────────────────────────────────┐
│  cli/          argument parsing · streaming render · approval    │
│                prompt · replay · cost                            │
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
examining are the ones that crashed. `--replay` re-narrates it — the interesting
question about a finished session is "what did it actually do", and a second run
would behave differently anyway.

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
the user for, not what is permitted; the approval mode is the real gate. It is not
a sandbox: a determined command still runs with the user's privileges. Container
isolation would be the next step.

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
