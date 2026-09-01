"""The agent loop: an explicit state machine, not a ``while True``.

One turn is a cycle. The model is asked what to do; if it answers in prose, the
turn is over; if it requests tools, each call is authorised, executed, and its
result appended, and the model is asked again with what it learned. That cycle
is the whole idea, and the value is in the edges rather than the happy path:

* **Every tool call gets a result.** Even a refused one, a malformed one, or a
  call to a tool that does not exist. Both wire formats reject a request whose
  ``tool_calls`` lack answers, so an unanswered call does not degrade the run —
  it ends it. Refusals and errors are therefore phrased as feedback the model can
  act on, which is also what makes self-correction possible at all.
* **Nothing a tool does can stop the loop.** ``BaseTool.invoke`` already converts
  failure into an outcome; the engine adds the same treatment for approval
  refusals and for its own dispatch errors.
* **Termination is decided here, not by the model.** A model that keeps calling
  tools forever is stopped by :class:`~cagent.agent.guards.LoopGuard`; a model
  that stops early ends the turn. Neither outcome is left to chance.

The engine holds no UI, no printing, and no input. It emits events and calls
injected callbacks, which is why it can be driven identically by the terminal
app and by the evaluation harness.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..config import AgentConfig
from ..errors import (
    CagentError,
    ContextOverflowError,
    ContextWindowTooSmall,
    LoopGuardError,
    ProviderError,
    ToolError,
    UserAbort,
)
from ..llm.base import LLMProvider
from ..llm.base import TextDelta as WireTextDelta
from ..llm.base import ThinkingDelta as WireThinkingDelta
from ..tools.base import ApprovalRequest, BaseTool, ToolContext, ToolOutcome
from ..tools.registry import ToolRegistry, default_registry
from ..tools.schema import parse_object
from ..types import Message, RiskLevel, TextPart, ToolCallPart, ToolResultPart, Usage
from .approval import ApprovalPolicy
from .context import ContextManager
from .events import (
    Activity,
    AgentEvent,
    ApprovalDecided,
    ApprovalRequested,
    CompactionDone,
    EventSink,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    UserMessage,
    Warning,
)
from .guards import LoopGuard
from .prompt import FOCUS_HEADER, PromptBuilder
from .sandbox import SandboxError, SandboxSession, docker_available, docker_image_available

__all__ = ["Agent", "TurnResult"]

_SUMMARY_INSTRUCTION = """\
Summarise the work so far for your own future reference. Include: what the task \
is, which files you have inspected or changed and how, what commands you ran and \
what they reported, what is verified, and what remains. Be specific about paths \
and symbols. Write it as notes to yourself, not as a report to a user. Do not \
use tools."""

_MALFORMED_ARGS_FEEDBACK = """\
The arguments for this call were not valid JSON, so it could not be run. Emit the \
call again with well-formed JSON arguments. If a value contains quotes, newlines, \
or backslashes, escape them properly."""

_REFUSED_FEEDBACK = """\
The user declined to run this. Do not retry it. Either continue with a different \
approach, or explain what you need permission for and stop."""

_STALE_MAP_NOTE = """\
[The project map in the system prompt was built before this turn and is not \
rebuilt mid-turn. Changed since: {changed}. Read a file rather than trusting \
the map's description of it.]"""

_MAX_LISTED_CHANGES = 8

_MAX_STALE_FOCUS_BLOCKS = 3
"""How many superseded task-focus blocks may accumulate before they are swept.

Sweeping rewrites the transcript from the earliest one onward, so doing it every
turn would trade a cached prefix for a few hundred tokens — the wrong way round.
Waiting until several have piled up makes the sweep worth its own cost."""

_MAX_PARALLEL_TOOLS = 4


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The outcome of one user turn."""

    reply: str
    steps: int
    usage: Usage
    stopped_by: str
    """``"model"`` when the model finished, otherwise the guard or error that
    ended the turn."""

    @property
    def completed(self) -> bool:
        return self.stopped_by == "model"


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    call: ToolCallPart
    tool: BaseTool
    context: ToolContext
    risk: RiskLevel
    nudge: str | None


def _changed_path(call: ToolCallPart) -> str | None:
    """The file a mutating call named, when it named one.

    File tools all take ``path``; ``run_bash`` names nothing, and guessing what
    a shell command touched would be worse than admitting it is unknown.
    """
    value = call.arguments.get("path")
    return value if isinstance(value, str) and value.strip() else None


@dataclass(slots=True)
class Agent:
    """Runs tasks against a provider using a set of tools.

    Constructed with its collaborators rather than building them, so a test can
    substitute a fake provider, a scripted approval policy, or a registry holding
    one tool. Only :meth:`create` reaches for the real ones.
    """

    config: AgentConfig
    provider: LLMProvider
    registry: ToolRegistry
    policy: ApprovalPolicy
    sink: EventSink
    defer_initial_prompt: bool = False
    context: ContextManager = field(init=False)
    prompt_builder: PromptBuilder = field(init=False)
    guard: LoopGuard = field(init=False)
    abort: threading.Event = field(default_factory=threading.Event)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    usage: Usage = field(default_factory=Usage)
    started: float = field(default_factory=time.monotonic)
    _system: str = ""
    _current_task: str = ""
    _files_changed: bool = False
    _pending_changes: set[str] = field(default_factory=set)
    """Paths mutated since the model was last told so. Drained into one note
    per step, rather than triggering a system-prompt rebuild."""
    _opaque_changes: bool = False
    """Whether a mutation happened that names no path, such as a shell command."""
    _initialized: bool = field(init=False, default=False)
    sandbox: SandboxSession | None = field(init=False, default=None)
    sandbox_warning: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.sandbox, self.sandbox_warning = SandboxSession.create_with_status(self.config)
        self.prompt_builder = PromptBuilder(
            self.config, workspace=self.sandbox.workspace if self.sandbox is not None else None
        )
        self.guard = LoopGuard(self.config)
        self.context = ContextManager(self.config, summarizer=self._summarise)
        if not self.defer_initial_prompt:
            self.initialize()

    @classmethod
    def create(
        cls,
        config: AgentConfig,
        *,
        sink: EventSink,
        policy: ApprovalPolicy | None = None,
        registry: ToolRegistry | None = None,
        provider: LLMProvider | None = None,
        defer_initial_prompt: bool = False,
    ) -> Agent:
        """Build an agent with the default provider, tools, and policy."""
        from ..llm.factory import build_provider

        return cls(
            config=config,
            provider=provider or build_provider(config),
            registry=registry or default_registry(),
            policy=policy or ApprovalPolicy(config),
            sink=sink,
            defer_initial_prompt=defer_initial_prompt,
        )

    # ---------------------------------------------------------------- lifecycle

    def _emit(self, event: AgentEvent) -> None:
        self.sink.handle(event)

    def initialize(self) -> None:
        """Build the initial system prompt once.

        The TUI defers this work until after its first paint so repo-map
        construction can be reported instead of freezing an empty terminal.
        Other frontends keep the default eager initialization.
        """
        if self._initialized:
            return
        self._refresh_system_prompt()
        self._initialized = True

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system prompt and re-cost the request overhead.

        Only ever called between turns. The system prompt is the provider's
        cache prefix, so rewriting it mid-turn discards the cached transcript
        at the point that transcript is longest; :meth:`_note_stale_map` covers
        the mid-turn case for a few tokens instead.
        """
        specs = tuple(self.registry.specs())
        prompt = self.prompt_builder.build(tools=specs)
        self._system = prompt.text
        self.context.set_overhead(system_tokens=prompt.tokens, tools=specs)

    def announce(self, task: str) -> None:
        """Emit the opening event for a session."""
        self.initialize()
        shell_access = "container" if self.sandbox is not None else "host (unrestricted)"
        path_boundary = "unrestricted" if self.config.allow_outside_workspace else "workspace-only"
        self._emit(
            RunStarted(
                task=task,
                model=self.config.resolved_model,
                endpoint=self.config.resolved_base_url,
                system_tokens=self.context.system_tokens,
                tool_names=tuple(sorted(self.registry.names())),
                sandbox_status=self.sandbox_startup_status(),
                shell_access=shell_access,
                path_boundary=path_boundary,
            )
        )
        if self.sandbox_warning is not None:
            self._emit(
                Warning(
                    "run_bash is using the host with unrestricted process access.",
                    detail=(
                        f"{self.sandbox_warning} File tools remain {path_boundary}; "
                        "a host shell can access outside the workspace via cd, absolute "
                        "paths, redirection, symlinks, or child processes. Use "
                        "sandbox_mode = 'docker' or /sandbox on for isolation."
                    ),
                )
            )

    def restore_history(self, messages: Sequence[Message]) -> None:
        """Replace the empty transcript with a validated resumed history.

        A resumed process is a new Agent: usage, loop limits, approvals, and
        sandbox state intentionally start fresh. Only the provider conversation
        is restored from the trace.
        """
        self.context.history = list(messages)
        self.context.compactions = 0
        self.guard.note_progress()

    def undo_last_turn(self) -> int:
        """Remove the latest user turn and everything that answered it.

        Tool calls and their results are deliberately removed together. This
        only changes model context; filesystem edits and command side effects
        from the turn have already happened and cannot be undone here.

        Returns:
            The number of messages removed, or zero when there is no user turn.
        """
        for index in range(len(self.context.history) - 1, -1, -1):
            message = self.context.history[index]
            if message.role != "user" or message.synthetic:
                continue
            removed = len(self.context.history) - index
            del self.context.history[index:]
            self.guard.note_progress()
            return removed
        return 0

    def finish(self, reason: str, *, trace_path: str | None = None) -> RunFinished:
        """Emit and return the closing event."""
        self._finalize_sandbox()
        event = RunFinished(
            reason=reason,
            steps=self.guard.steps,
            usage=self.usage,
            elapsed_s=time.monotonic() - self.started,
            trace_path=trace_path,
        )
        self._emit(event)
        return event

    # -------------------------------------------------------------------- turns

    def run_turn(self, task: str) -> TurnResult:
        """Work one user turn to completion.

        Args:
            task: What the user asked for.

        Returns:
            The reply and how the turn ended. Errors are reported here rather
            than raised: a provider failure or a spent budget is information for
            the user, and the session survives it.
        """
        if not self._initialized:
            self._emit(Activity(self._initialization_activity()))
            self.initialize()
        self._current_task = task
        prefix_cold = self._prepare_task_context(task)
        self.context.append(Message.user(task))
        self._emit(UserMessage(task))
        self._attach_task_focus(task, prefix_cold=prefix_cold)
        self.guard.note_progress()

        turn_start_usage = self.usage
        steps_at_start = self.guard.steps
        reply = ""
        stopped_by = "model"

        try:
            reply, stopped_by = self._loop()
        except LoopGuardError as exc:
            stopped_by = type(exc).__name__
            reply = self._explain_stop(exc)
            self._emit(Warning(reply))
        except UserAbort as exc:
            stopped_by = "aborted"
            reply = str(exc) or "Interrupted."
            self._emit(Warning(reply))
        except ContextWindowTooSmall as exc:
            stopped_by = "context_window"
            reply = str(exc)
            self._emit(Warning(reply))
        except ProviderError as exc:
            stopped_by = "provider_error"
            reply = f"The model could not be reached: {exc}"
            self._emit(Warning(reply, detail=type(exc).__name__))
        except CagentError as exc:
            stopped_by = "error"
            reply = f"{type(exc).__name__}: {exc}"
            self._emit(Warning(reply))

        result = TurnResult(
            reply=reply,
            steps=self.guard.steps - steps_at_start,
            usage=self._usage_delta(turn_start_usage),
            stopped_by=stopped_by,
        )
        self._emit(TurnFinished(reply=reply, steps=result.steps, usage=result.usage))
        return result

    def _loop(self) -> tuple[str, str]:
        """Alternate between asking the model and running what it asked for.

        Returns the final prose and the reason the turn stopped.
        """
        overflow_retries = 0
        while True:
            if self.abort.is_set():
                raise UserAbort("Interrupted.")

            self._compact_if_needed()
            self.guard.before_step()

            step = self.guard.steps
            self._emit(Activity("Waiting for agent response"))
            self._emit(
                StepStarted(
                    step=step,
                    prompt_tokens_estimate=self.context.token_count(),
                )
            )

            try:
                result = self.provider.complete(
                    self.context.history,
                    system=self._system,
                    tools=self.registry.specs(),
                    abort=self.abort,
                    on_event=self._forward_stream_event,
                )
            except ContextOverflowError as exc:
                # Token estimation is intentionally conservative but cannot
                # know every provider's hidden prompt overhead. Give one
                # provider-reported overflow a fresh three-stage compaction
                # attempt; if that still cannot fit, stop with an actionable
                # window-size message instead of retrying the same request.
                if self.abort.is_set():
                    raise UserAbort("Interrupted.") from exc
                if overflow_retries:
                    raise ContextWindowTooSmall(
                        "The model rejected the context after compaction "
                        f"({exc}). Increase --context-window and retry, or start a new session."
                    ) from exc
                overflow_retries += 1
                report = self.context.compact()
                current_tokens = self.context.token_count()
                if report.changed:
                    self._emit(
                        CompactionDone(
                            strategy=report.strategy,
                            tokens_before=report.tokens_before,
                            tokens_after=report.tokens_after,
                            messages_before=report.messages_before,
                            messages_after=report.messages_after,
                        )
                    )
                if not report.changed or current_tokens > self.config.context_window:
                    raise ContextWindowTooSmall(
                        "The model rejected the context and the three compaction stages "
                        "could not reduce it below the configured window "
                        f"({self.config.context_window:,} tokens). "
                        "Increase --context-window and retry, or start a new session."
                    ) from exc
                continue

            overflow_retries = 0
            self.usage = self.usage + result.usage
            self.guard.add_tokens(result.usage.total)
            self.context.append(result.message)
            self._emit(
                StepFinished(
                    step=step,
                    message=result.message,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    latency_s=result.latency_s,
                )
            )

            calls = result.message.tool_calls
            if result.finish_reason in ("aborted", "length", "content_filter", "error"):
                # A stopped stream may contain a partial tool call. Do not run it
                # or retain an assistant turn that providers would require us to
                # answer with a tool result.
                if calls:
                    self.context.history.pop()
                if result.finish_reason == "aborted":
                    raise UserAbort("Interrupted.")
                messages = {
                    "length": "The model's reply hit the output token limit and may be incomplete.",
                    "content_filter": (
                        "The provider stopped the model's response with a content filter; "
                        "the reply may be incomplete."
                    ),
                    "error": (
                        "The provider ended the response with an error; "
                        "the reply may be incomplete."
                    ),
                }
                self._emit(Warning(messages[result.finish_reason]))
                return result.message.text, result.finish_reason

            if not calls:
                return result.message.text, (
                    "model" if result.finish_reason == "stop" else result.finish_reason
                )

            results, dispatch_error = self._dispatch(calls)
            self.context.append(Message.from_tool_results(results))

            if dispatch_error is not None:
                raise dispatch_error

            if self.policy.aborted:
                raise UserAbort("Stopped at your request.")

    def _forward_stream_event(self, event: object) -> None:
        """Republish provider deltas as engine events, so the UI can stream."""
        if isinstance(event, WireTextDelta):
            self._emit(TextDelta(event.text))
        elif isinstance(event, WireThinkingDelta) and event.text:
            self._emit(ThinkingDelta(event.text))

    # ------------------------------------------------------------------- tools

    def _dispatch(
        self, calls: Sequence[ToolCallPart]
    ) -> tuple[list[ToolResultPart], LoopGuardError | None]:
        """Execute a batch, returning one result per call and any stop reason.

        The one-result-per-call guarantee is what keeps the transcript valid, so
        every early exit in here still produces a result part. A loop-guard
        failure is returned after the results are assembled, allowing the caller
        to append the complete tool turn before stopping.
        """
        if len(calls) > 1 and self._parallel_safe(calls):
            return self._dispatch_parallel(calls)

        results: list[ToolResultPart] = []
        for index, call in enumerate(calls):
            if self.abort.is_set() or self.policy.aborted:
                reason = (
                    "the user interrupted"
                    if self.abort.is_set()
                    else "the user stopped the turn"
                )
                results.append(
                    ToolResultPart(call.id, f"Not run: {reason}.", is_error=True)
                )
                continue
            try:
                results.append(self._run_one(call))
            except LoopGuardError as exc:
                results.append(ToolResultPart(call.id, f"Not run: {exc}", is_error=True))
                results.extend(
                    ToolResultPart(
                        pending.id,
                        "Not run: the loop guard stopped this turn.",
                        is_error=True,
                    )
                    for pending in calls[index + 1 :]
                )
                return results, exc
        return results, None

    def _parallel_safe(self, calls: Sequence[ToolCallPart]) -> bool:
        """Whether every call explicitly opts into read-only concurrency."""
        try:
            tools = [self.registry.get(call.name) for call in calls]
        except CagentError:
            return False
        return all(tool.parallel_safe and tool.risk is RiskLevel.SAFE for tool in tools)

    def _dispatch_parallel(
        self, calls: Sequence[ToolCallPart]
    ) -> tuple[list[ToolResultPart], LoopGuardError | None]:
        """Prepare serially, invoke concurrently, and preserve result order."""
        slots: list[ToolResultPart | _PreparedCall] = []
        for index, call in enumerate(calls):
            if self.abort.is_set() or self.policy.aborted:
                reason = (
                    "the user interrupted"
                    if self.abort.is_set()
                    else "the user stopped the turn"
                )
                slots.append(ToolResultPart(call.id, f"Not run: {reason}.", is_error=True))
                continue
            try:
                slots.append(self._prepare_call(call))
            except LoopGuardError as exc:
                results = [
                    slot
                    if isinstance(slot, ToolResultPart)
                    else ToolResultPart(
                        slot.call.id,
                        "Not run: the loop guard stopped this turn.",
                        is_error=True,
                    )
                    for slot in slots
                ]
                results.append(ToolResultPart(call.id, f"Not run: {exc}", is_error=True))
                results.extend(
                    ToolResultPart(
                        pending.id,
                        "Not run: the loop guard stopped this turn.",
                        is_error=True,
                    )
                    for pending in calls[index + 1 :]
                )
                return results, exc

        prepared = [slot for slot in slots if isinstance(slot, _PreparedCall)]
        if any(item.risk is not RiskLevel.SAFE for item in prepared):
            return [
                slot if isinstance(slot, ToolResultPart) else self._execute_call(slot)
                for slot in slots
            ], None

        for item in prepared:
            self._emit(ToolStarted(call=item.call, risk=item.risk))

        completed: list[tuple[ToolOutcome, float]] = []
        if prepared:
            with ThreadPoolExecutor(max_workers=min(len(prepared), _MAX_PARALLEL_TOOLS)) as pool:
                completed = list(pool.map(self._invoke_call, prepared))

        completed_iter = iter(completed)
        results = [
            slot
            if isinstance(slot, ToolResultPart)
            else self._finish_call(slot, *next(completed_iter))
            for slot in slots
        ]
        return results, None

    def _run_one(self, call: ToolCallPart) -> ToolResultPart:
        """Authorise and execute a single call."""
        prepared = self._prepare_call(call)
        if isinstance(prepared, ToolResultPart):
            return prepared
        return self._execute_call(prepared)

    def _execute_call(self, prepared: _PreparedCall) -> ToolResultPart:
        """Execute one prepared call synchronously."""
        self._emit(ToolStarted(call=prepared.call, risk=prepared.risk))
        return self._finish_call(prepared, *self._invoke_call(prepared))

    def _prepare_call(self, call: ToolCallPart) -> ToolResultPart | _PreparedCall:
        """Validate and authorise a call without executing it."""
        if call.raw_arguments and not call.arguments:
            return ToolResultPart(call.id, _MALFORMED_ARGS_FEEDBACK, is_error=True)

        try:
            tool = self.registry.get(call.name)
        except ToolError as exc:
            # An unknown tool is a model mistake with an obvious repair, so the
            # feedback carries the error's hint — the names that do exist —
            # rather than just the message.
            return ToolResultPart(call.id, exc.as_model_feedback(), is_error=True)
        except CagentError as exc:
            return ToolResultPart(call.id, str(exc), is_error=True)

        nudge = self.guard.check_call(call)
        ctx = self._tool_context()
        parsed, request = self._plan_call(tool, call, ctx)

        # Arguments that do not parse are handled by invoke(), which produces the
        # canonical message. Skipping approval for them is safe precisely because
        # such a call cannot reach the tool's side effects.
        if parsed and not self._authorise(request):
            return ToolResultPart(call.id, _REFUSED_FEEDBACK, is_error=True)

        risk = request.risk if request is not None else tool.risk
        return _PreparedCall(call, tool, ctx, risk, nudge)

    @staticmethod
    def _invoke_call(prepared: _PreparedCall) -> tuple[ToolOutcome, float]:
        """Run only the tool body; safe to call from a worker thread."""
        started = time.perf_counter()
        outcome = prepared.tool.invoke(prepared.call.arguments, prepared.context)
        duration = time.perf_counter() - started
        return outcome, duration

    def _finish_call(
        self, prepared: _PreparedCall, outcome: ToolOutcome, duration: float
    ) -> ToolResultPart:
        """Publish one completed invocation and build its model-facing result."""
        self._emit(ToolFinished(call=prepared.call, outcome=outcome, duration_s=duration))

        if not outcome.is_error and prepared.risk >= RiskLevel.MUTATING:
            # The repo map described the tree as it was; a mutation makes it
            # stale. The rescan waits for the turn boundary so the cached
            # prompt prefix survives; the model is told in the meantime.
            self._files_changed = True
            changed = _changed_path(prepared.call)
            if changed is None:
                self._opaque_changes = True
            else:
                self._pending_changes.add(changed)

        content = outcome.content
        if prepared.nudge:
            content = f"{content}\n\n{prepared.nudge}"
        return ToolResultPart(prepared.call.id, content, is_error=outcome.is_error)

    @staticmethod
    def _plan_call(
        tool: BaseTool, call: ToolCallPart, ctx: ToolContext
    ) -> tuple[bool, ApprovalRequest | None]:
        """Parse arguments and ask the tool what permission it needs.

        Returns whether the arguments were usable, and the approval request if
        the tool produced one. A tool that raises while describing itself — a
        dry-run diff against a file that vanished, say — yields no request rather
        than failing the call, and the real error surfaces from the tool's own
        execution where it can be reported properly.
        """
        params: Any
        try:
            params = parse_object(tool.Params, call.arguments)
        except CagentError:
            return False, None

        try:
            return True, tool.approval_request(params, ctx)
        except Exception:  # noqa: BLE001  # describing a call must not stop the run
            return True, None

    def _authorise(self, request: ApprovalRequest | None) -> bool:
        """Run one request past the policy, reporting both sides to the UI."""
        if request is None:
            return self.policy.decide(None).approved

        will_ask = self.policy.requires_prompt(request)
        if will_ask:
            self._emit(ApprovalRequested(request))

        decision = self.policy.decide(request)
        self._emit(
            ApprovalDecided(
                request=request,
                approved=decision.approved,
                remembered=decision.remember,
                automatic=not will_ask,
            )
        )
        return decision.approved

    def _tool_context(self) -> ToolContext:
        """Fresh context per call, so a tool cannot retain host state."""
        workspace = self.sandbox.workspace if self.sandbox is not None else self.config.workspace
        return ToolContext(
            workspace=workspace,
            config=self.config,
            approve=lambda request: self.policy.decide(request).approved,
            emit=lambda line: self._emit(Warning(line)),
            abort=self.abort,
            force_workspace_boundary=self.sandbox is not None,
            sandbox=self.sandbox,
        )

    # ------------------------------------------------------------ housekeeping

    def _compact_if_needed(self) -> None:
        """Compact history before a request that would otherwise overflow."""
        self._note_stale_map()

        if not self.context.needs_compaction():
            return

        report = self.context.compact()
        current_tokens = self.context.token_count()
        if report.changed:
            self._emit(
                CompactionDone(
                    strategy=report.strategy,
                    tokens_before=report.tokens_before,
                    tokens_after=report.tokens_after,
                    messages_before=report.messages_before,
                    messages_after=report.messages_after,
                )
            )
        if current_tokens <= self.config.context_window:
            return
        message = (
            "The context window is too small: history remains at "
            f"{self.context.token_count():,} estimated tokens after all three "
            f"compaction stages (window {self.config.context_window:,}). Increase "
            "--context-window and retry, or start a new session."
        )
        raise ContextWindowTooSmall(message)

    def _prepare_task_context(self, task: str) -> bool:
        """Rebuild a map the previous turn made stale.

        Returns:
            Whether the system prompt changed, and with it whether the
            provider's cached prefix for this session is already cold.
        """
        if self._files_changed:
            self._emit(Activity(self._refresh_activity()))
            self.prompt_builder.invalidate_map()
            self._refresh_system_prompt()
            self._files_changed = False
            self._pending_changes.clear()
            self._opaque_changes = False
            return True
        activity = (
            "Preparing task context" if self.config.repo_map_enabled else "Preparing context"
        )
        self._emit(Activity(activity))
        return False

    def _attach_task_focus(self, task: str, *, prefix_cold: bool) -> None:
        """Append the task-ranked slice of the map after the user's turn.

        Here rather than in the system prompt because it is the one part of the
        map that changes every turn. Appending leaves the whole cached prefix —
        tool schemas, system prompt, and the transcript so far — untouched,
        which is the difference between paying for the map and paying for the
        map plus a re-read of the entire conversation.
        """
        self._retire_previous_focus(forced=prefix_cold)
        focus = self.prompt_builder.task_focus(task)
        if focus is None:
            return
        self.context.append(Message(role="user", parts=[TextPart(focus)], synthetic=True))

    def _retire_previous_focus(self, *, forced: bool) -> None:
        """Drop earlier turns' focus blocks, but not at any price.

        Each is derived context for a task that is over, and a session of a
        dozen turns would otherwise carry a dozen of them. Removing one is safe
        at any point — a standalone user message with no tool call to orphan —
        but it is not free: the earliest surviving block sits near the front of
        the transcript, so deleting it invalidates the cached prefix from there
        on, which costs far more than the stale tokens it reclaims.

        So they go when the prefix is cold anyway (the map was rebuilt, so the
        system prompt already changed), or when enough have piled up that the
        one-off re-read is finally the cheaper side of the trade. In practice
        the first condition fires on any turn that wrote a file, which is most
        of them.
        """
        stale = [
            index
            for index, message in enumerate(self.context.history)
            if message.synthetic
            and message.role == "user"
            and message.text.startswith(FOCUS_HEADER)
        ]
        if not stale or (not forced and len(stale) < _MAX_STALE_FOCUS_BLOCKS):
            return
        drop = set(stale)
        self.context.history = [
            message
            for index, message in enumerate(self.context.history)
            if index not in drop
        ]

    def _note_stale_map(self) -> None:
        """Tell the model the map aged, instead of rebuilding the system prompt.

        A mid-turn rebuild is correct and ruinously expensive: it rewrites the
        cached prefix at the moment the transcript is longest, so the provider
        re-reads everything. The information the rebuild carried is small — some
        files are no longer as described — and an appended line carries it
        without moving the prefix. The real rebuild happens at the next turn
        boundary, where the cache would have been cold anyway.
        """
        if not self._files_changed:
            return
        changed = sorted(self._pending_changes)
        self._pending_changes.clear()
        opaque = self._opaque_changes
        self._opaque_changes = False
        if not changed and not opaque:
            # Already reported; the map stays flagged for the turn boundary.
            return

        listed = ", ".join(changed[:_MAX_LISTED_CHANGES])
        if len(changed) > _MAX_LISTED_CHANGES:
            listed += f", and {len(changed) - _MAX_LISTED_CHANGES} more"
        if opaque:
            listed = f"{listed}, plus other tool side effects" if listed else "unnamed files"
        self.context.append(
            Message(
                role="user",
                parts=[TextPart(_STALE_MAP_NOTE.format(changed=listed))],
                synthetic=True,
            )
        )

    def _initialization_activity(self) -> str:
        return "Building repo map" if self.config.repo_map_enabled else "Preparing context"

    def _refresh_activity(self) -> str:
        return "Refreshing repo map" if self.config.repo_map_enabled else "Preparing context"

    def _summarise(self, messages: Sequence[Message]) -> str:
        """Condense history with a separate, tool-free model call.

        Deliberately not part of the main loop: it runs without tools, without
        the repo map, and its cost is counted against the run's usage so a
        session's total reflects everything it spent.
        """
        request = [*messages, Message.user(_SUMMARY_INSTRUCTION)]
        result = self.provider.complete(request, system="", tools=(), abort=self.abort)
        self.usage = self.usage + result.usage
        self.guard.add_tokens(result.usage.total)
        return result.message.text

    def _usage_delta(self, before: Usage) -> Usage:
        """Tokens spent since ``before``."""
        return Usage(
            prompt_tokens=self.usage.prompt_tokens - before.prompt_tokens,
            completion_tokens=self.usage.completion_tokens - before.completion_tokens,
            cached_tokens=self.usage.cached_tokens - before.cached_tokens,
            reasoning_tokens=self.usage.reasoning_tokens - before.reasoning_tokens,
        )

    @staticmethod
    def _explain_stop(exc: LoopGuardError) -> str:
        """Turn a guard trip into something a user can act on."""
        from ..errors import RepetitionDetected, TokenBudgetExceeded

        match exc:
            case TokenBudgetExceeded():
                return f"Stopped on the token budget ({exc}). Raise --token-budget to continue."
            case RepetitionDetected():
                return (
                    f"Stopped because the model repeated the same call without progress "
                    f"({exc}). It is likely missing information the tools cannot supply."
                )
            case _:
                return f"Stopped: {exc}"

    def interrupt(self) -> None:
        """Ask the run to stop at the next safe point."""
        self.abort.set()
        # Wake a provider blocked on network I/O immediately. The provider
        # still emits an ``aborted`` completion so the partial turn is saved.
        with contextlib.suppress(Exception):
            self.provider.cancel()

    def reset_interrupt(self) -> None:
        """Clear a previous interrupt so the session can continue."""
        self.abort.clear()
        self.policy.aborted = False

    def sandbox_status(self) -> str:
        """Return a concise status line for the interactive CLI."""
        if self.sandbox is None:
            if self.config.sandbox_mode == "auto":
                return f"auto (host fallback; sync: {self.config.sandbox_sync})"
            return "off"
        container = self.sandbox.container_name or "not started"
        network = "bridge" if self.config.sandbox_network else "none"
        return (
            f"docker (container: {container}; sync: {self.config.sandbox_sync}; "
            f"network: {network})"
        )

    def sandbox_startup_status(self) -> str:
        """Return a stable sandbox label for the non-refreshing startup panel."""
        if self.sandbox is not None:
            return "docker"
        return self.sandbox_status()

    def enable_sandbox(self, *, image: str | None = None) -> None:
        """Create a disposable snapshot and enable Docker execution."""
        if self.sandbox is not None:
            if image is not None:
                self.set_sandbox_image(image)
            return
        previous_mode = self.config.sandbox_mode
        previous_image = self.config.sandbox_image
        if image is not None:
            if not image.strip():
                raise SandboxError("Sandbox image must not be empty.")
            self.config.sandbox_image = image.strip()
        self.config.sandbox_mode = "docker"
        try:
            sandbox = SandboxSession.create(self.config)
        except SandboxError:
            self.config.sandbox_mode = previous_mode
            self.config.sandbox_image = previous_image
            raise
        assert sandbox is not None
        self.sandbox = sandbox
        self.sandbox_warning = None
        self.prompt_builder.workspace = sandbox.workspace
        self.prompt_builder.invalidate_map()
        self._emit(Activity(self._refresh_activity()))
        self._refresh_system_prompt()

    def set_sandbox_image(self, image: str) -> None:
        """Select an image, recycling only the container and keeping the snapshot.

        The image is checked before anything is changed. An unavailable name used
        to be accepted here and only surfaced from ``ensure_container`` on the next
        ``run_bash``, leaving the file tools writing into a snapshot no command
        could run in.
        """
        image = image.strip()
        if not image:
            raise SandboxError("Sandbox image must not be empty.")
        if image == self.config.sandbox_image:
            return
        if not docker_available():
            raise SandboxError(
                "Cannot select a sandbox image: the Docker CLI or daemon is unavailable."
            )
        if not docker_image_available(image):
            raise SandboxError(
                f"Sandbox image {image!r} is not available locally; cagent never pulls "
                f"images automatically. Still using {self.config.sandbox_image!r}."
            )
        if self.sandbox is not None:
            self.sandbox.stop_container()
        self.config.sandbox_image = image

    def disable_sandbox(self) -> None:
        """Synchronise/discard the snapshot and return tools to the host tree."""
        sandbox = self.sandbox
        if sandbox is None:
            self.config.sandbox_mode = "off"
            self.sandbox_warning = "Docker sandboxing was explicitly disabled."
            return
        try:
            self._finish_sandbox()
        finally:
            sandbox.close()
            self.sandbox = None
            self.config.sandbox_mode = "off"
            self.sandbox_warning = "Docker sandboxing was explicitly disabled."
            self.prompt_builder.workspace = None
            self.prompt_builder.invalidate_map()
            self._emit(Activity(self._refresh_activity()))
            self._refresh_system_prompt()

    def apply_sandbox_changes(self) -> tuple[str, ...]:
        """Immediately sync current sandbox edits while keeping it enabled."""
        if self.sandbox is None:
            raise SandboxError("Sandbox is not active.")
        changed = self.sandbox.changed_paths
        if not changed:
            self._emit(Warning("No pending sandbox changes."))
            return ()
        applied = self.sandbox.apply()
        self._emit(Warning(f"Copied {len(applied)} sandbox change(s) back to the project."))
        self.prompt_builder.invalidate_map()
        self._emit(Activity(self._refresh_activity()))
        self._refresh_system_prompt()
        return applied

    def discard_sandbox_changes(self) -> None:
        """Immediately discard current sandbox edits while keeping it enabled."""
        if self.sandbox is None:
            raise SandboxError("Sandbox is not active.")
        changed = self.sandbox.changed_paths
        self.sandbox.discard_changes()
        if changed:
            self._emit(Warning(f"Discarded {len(changed)} sandbox change(s)."))
        else:
            self._emit(Warning("No pending sandbox changes."))
        self.prompt_builder.invalidate_map()
        self._emit(Activity(self._refresh_activity()))
        self._refresh_system_prompt()

    def close(self) -> None:
        """Synchronise or discard an isolated workspace, then close the provider."""
        self._finalize_sandbox()
        self.provider.close()

    def _finalize_sandbox(self) -> None:
        """Finish a sandbox exactly once, without letting cleanup break shutdown."""
        sandbox = self.sandbox
        if sandbox is None:
            return
        try:
            self._finish_sandbox()
        except SandboxError as exc:
            self._emit(Warning("Sandbox changes were not copied back.", detail=str(exc)))
        finally:
            sandbox.close()
            self.sandbox = None

    def _finish_sandbox(self) -> None:
        """Handle the explicit boundary from a disposable copy to the project."""
        assert self.sandbox is not None
        changed = self.sandbox.changed_paths
        if not changed:
            return
        if self.config.sandbox_sync == "never":
            self._emit(Warning("Sandbox changes were discarded; the project was left untouched."))
            return
        request = ApprovalRequest(
            tool="sandbox_sync",
            risk=RiskLevel.MUTATING,
            summary=f"copy {len(changed)} sandbox change(s) back to the project",
            detail=self.sandbox.diff(),
            signature="sandbox_sync",
            always_prompt=self.config.sandbox_sync == "ask",
        )
        if self.config.sandbox_sync == "always" or self._authorise(request):
            try:
                applied = self.sandbox.apply()
            except SandboxError as exc:
                self._emit(Warning("Sandbox changes were not copied back.", detail=str(exc)))
            else:
                self._emit(Warning(f"Copied {len(applied)} sandbox change(s) back to the project."))
        else:
            self._emit(Warning("Sandbox changes were discarded; the project was left untouched."))
