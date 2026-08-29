"""Keeping the transcript inside the context window.

Every step adds an assistant turn and a batch of tool results, so a long task
grows its own prompt until the request is rejected. Something has to remove
history, and what it removes decides whether the agent stays coherent.

Two properties constrain any solution here.

**Tool calls and their results are inseparable.** Both wire protocols reject a
request containing an assistant turn whose ``tool_calls`` have no matching
results, or results with no originating call. So history cannot be trimmed
message by message; it is segmented into blocks — a user turn plus all the
model and tool activity answering it — and blocks are the unit of removal. This
is the invariant most likely to be violated by a naive sliding window, and it
fails as a provider 400 rather than as degraded output.

**Not all history is equally valuable.** The bulk is tool output, and old tool
output is the least useful text in the window: a file that has since been edited,
a test run that has since been re-run. The original task, by contrast, must
survive to the last step — an agent that forgets what it was asked will confidently
finish the wrong job.

So compaction proceeds in escalating stages, cheapest first: elide old tool
output, then summarise old blocks into a progress note, then drop them. Each
stage reports what it did, because silently forgetting context looks to the user
like the model getting confused.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..config import AgentConfig
from ..llm.tokens import estimate_message, estimate_messages
from ..tools.truncation import truncate_output
from ..types import Message, TextPart, ToolResultPart, ToolSpec

__all__ = ["Block", "CompactionReport", "ContextManager", "Summarizer"]

Summarizer = Callable[[Sequence[Message]], str]
"""Condenses a run of history into a progress note. Supplied by the engine, so
this module never talks to a provider itself."""

_ELIDED_TOOL_OUTPUT = "[earlier tool output removed to save context]"

_ELIDE_KEEP_HEAD_LINES = 3
_ELIDE_KEEP_TAIL_LINES = 2
_ELIDE_MAX_CHARS = 400
"""How much of an old tool result survives elision. Enough to recall what the
call was about, not enough to re-read its content."""

_SUMMARY_HEADER = "Progress so far (earlier history was summarised to save context):"


@dataclass(slots=True)
class Block:
    """A user turn and everything that answered it.

    Removal happens a whole block at a time, which is what guarantees an
    assistant turn is never separated from its tool results.
    """

    messages: list[Message] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def tokens(self, model: str) -> int:
        return sum(estimate_message(message, model=model) for message in self.messages)


Stage = Literal["elide", "summarise", "drop"]
"""One rung of the compaction ladder."""


@dataclass(frozen=True, slots=True)
class CompactionReport:
    """What one compaction pass accomplished."""

    stages: tuple[Stage, ...]
    """Every rung applied, in order. More than one means the cheaper stage did
    not free enough and the pass had to escalate."""

    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int

    @property
    def strategy(self) -> str:
        """The stages as one label, e.g. ``"elide+drop"``, or ``"none"``."""
        return "+".join(self.stages) if self.stages else "none"

    @property
    def changed(self) -> bool:
        return bool(self.stages)

    @property
    def saved(self) -> int:
        return max(self.tokens_before - self.tokens_after, 0)


@dataclass(slots=True)
class ContextManager:
    """Owns the transcript and keeps it affordable.

    The system prompt and tool schemas are not stored here but are counted, since
    they occupy the same window and a large repo map meaningfully shifts when
    compaction has to start.
    """

    config: AgentConfig
    history: list[Message] = field(default_factory=list)
    system_tokens: int = 0
    tool_tokens: int = 0
    summarizer: Summarizer | None = None
    compactions: int = 0

    def append(self, message: Message) -> None:
        """Add a message to the transcript."""
        self.history.append(message)

    def extend(self, messages: Sequence[Message]) -> None:
        self.history.extend(messages)

    def set_overhead(self, *, system_tokens: int, tools: Sequence[ToolSpec] = ()) -> None:
        """Record the fixed per-request cost outside the transcript."""
        from ..llm.tokens import estimate_tools

        self.system_tokens = system_tokens
        self.tool_tokens = estimate_tools(tools, model=self.config.model_for_tokens) if tools else 0

    def token_count(self) -> int:
        """Estimated tokens for the next request, overhead included."""
        model = self.config.model_for_tokens
        return self.system_tokens + self.tool_tokens + estimate_messages(
            self.history, model=model
        )

    @property
    def pressure(self) -> float:
        """Window usage as a fraction, for display."""
        window = max(self.config.context_window, 1)
        return self.token_count() / window

    def needs_compaction(self) -> bool:
        """Whether the next request would cross the compaction threshold."""
        return self.token_count() > self.config.compact_at_tokens

    def compact(self) -> CompactionReport:
        """Free context, escalating only as far as necessary.

        Returns:
            What was done. ``strategy == "none"`` means nothing could be freed —
            the transcript is already at its floor. The engine reports an
            actionable context-window error instead of sending a request that
            is certain to be rejected.
        """
        tokens_before = self.token_count()
        messages_before = len(self.history)
        target = self.config.compact_at_tokens
        applied: list[Stage] = []

        ladder: tuple[tuple[Stage, Callable[[], bool]], ...] = (
            ("elide", self._elide_old_tool_output),
            ("summarise", self._summarise_old_blocks),
            ("drop", self._drop_old_blocks),
        )
        for stage, attempt in ladder:
            if not attempt():
                continue
            applied.append(stage)
            self.compactions += 1
            # Keep a useful progress note when it fits the actual model window,
            # even if it remains above the preferred compaction threshold. If
            # it still exceeds the real window, continue to the final drop rung.
            if stage == "summarise" and self.token_count() <= self.config.context_window:
                break
            if self.token_count() <= target:
                break

        return CompactionReport(
            stages=tuple(applied),
            tokens_before=tokens_before,
            tokens_after=self.token_count(),
            messages_before=messages_before,
            messages_after=len(self.history),
        )

    def blocks(self) -> list[Block]:
        """Segment history into removable units.

        A block is one *step*: an assistant turn together with the tool results
        answering it. Tool messages attach to the assistant turn before them,
        which is the whole point — those two are what the wire formats refuse to
        see separated, so they can only be removed together. A user turn forms a
        block of its own.

        Segmenting per step rather than per user turn matters more than it
        sounds: an agentic task is one user message followed by dozens of steps,
        so grouping by user turn would produce a single indivisible block and
        compaction could never free anything during exactly the long task that
        needs it.
        """
        blocks: list[Block] = []
        current = Block()
        for message in self.history:
            starts_block = message.role in ("user", "assistant", "system")
            if starts_block and current.messages:
                blocks.append(current)
                current = Block()
            current.messages.append(message)
        if not current.is_empty:
            blocks.append(current)
        return blocks

    def _protected_split(self) -> tuple[list[Block], list[Block]]:
        """Split blocks into (compactable, protected).

        The first block is protected because it holds the task, and the last
        ``keep_recent_turns`` are protected because they hold the state the model
        is currently reasoning about.
        """
        blocks = self.blocks()
        keep_recent = max(self.config.keep_recent_turns, 1)
        if len(blocks) <= keep_recent + 1:
            return [], blocks
        return blocks[1 : len(blocks) - keep_recent], [blocks[0], *blocks[-keep_recent:]]

    def _rebuild(self, blocks: Sequence[Block]) -> None:
        """Replace history with the messages of ``blocks``, in order."""
        self.history = [message for block in blocks for message in block.messages]

    def _elide_old_tool_output(self) -> bool:
        """Shrink tool results outside the protected window.

        The result parts stay in place with shortened content, so every call
        keeps its answer and the wire format stays valid. This is the cheapest
        stage and usually the only one needed, because tool output dominates.

        The configured recent window is a preference, not a reason to leave the
        request over budget. If eliding the strictly old blocks is insufficient,
        progressively elide the oldest blocks in that recent window while
        preserving the latest block verbatim. This keeps the state most useful
        for the next request and avoids an unnecessary drop/summarise escalation
        for a transcript that is only slightly over the target.
        """
        compactable, protected = self._protected_split()
        if not compactable and len(protected) <= 2:
            return False

        changed = False
        # Oldest history is always the first thing to elide.
        candidates = list(compactable)
        # If that was not enough, consume the older part of the protected
        # recent window. The first block is the original task and the final
        # block is the state immediately preceding the next model request.
        candidates.extend(protected[1:-1])

        for block in candidates:
            for message in block.messages:
                if message.role != "tool":
                    continue
                new_parts: list[ToolResultPart] = []
                message_changed = False
                for part in message.tool_results:
                    shortened = self._elide(part.content)
                    if shortened == part.content:
                        new_parts.append(part)
                        continue
                    new_parts.append(
                        ToolResultPart(
                            call_id=part.call_id,
                            content=shortened,
                            is_error=part.is_error,
                        )
                    )
                    changed = True
                    message_changed = True
                if message_changed:
                    message.parts = list(new_parts)
                    message.token_estimate = None
            if self.token_count() <= self.config.compact_at_tokens:
                break
        return changed

    @staticmethod
    def _elide(content: str) -> str:
        """Reduce one tool result to a reminder of what it was."""
        if len(content) <= _ELIDE_MAX_CHARS:
            return content
        head, _ = truncate_output(
            content,
            head_lines=_ELIDE_KEEP_HEAD_LINES,
            tail_lines=_ELIDE_KEEP_TAIL_LINES,
            max_chars=_ELIDE_MAX_CHARS,
        )
        return f"{head}\n{_ELIDED_TOOL_OUTPUT}"

    def _summarise_old_blocks(self) -> bool:
        """Replace old blocks with a model-written progress note.

        Skipped when no summarizer is configured, or when the note would not be
        smaller than what it replaces — paying for a summarisation call that
        frees nothing is worse than dropping the blocks outright.
        """
        if self.summarizer is None:
            return False
        compactable, protected = self._protected_split()
        if not compactable:
            return False

        stale = [message for block in compactable for message in block.messages]
        original_tokens = sum(
            estimate_message(message, model=self.config.model_for_tokens) for message in stale
        )
        try:
            summary = self.summarizer(stale)
        except Exception:  # noqa: BLE001  # a failed summary falls back to dropping
            return False
        if not summary.strip():
            return False

        note = Message(
            role="user",
            parts=[TextPart(f"{_SUMMARY_HEADER}\n{summary.strip()}")],
            synthetic=True,
        )
        if estimate_message(note, model=self.config.model_for_tokens) >= original_tokens:
            return False

        first, *recent = protected
        self._rebuild([first, Block([note]), *recent])
        return True

    def _drop_old_blocks(self) -> bool:
        """Discard old blocks, leaving a marker that history was lost.

        The last resort. The marker matters: without it the model sees a jump
        from the task straight to recent tool output and re-does finished work.
        """
        compactable, protected = self._protected_split()
        if not compactable:
            return False

        dropped_messages = sum(len(block.messages) for block in compactable)
        marker = Message(
            role="user",
            parts=[
                TextPart(
                    f"[{dropped_messages} earlier messages were dropped to fit the context "
                    "window. Re-read any file you need instead of relying on memory of it.]"
                )
            ],
            synthetic=True,
        )
        first, *recent = protected
        self._rebuild([first, Block([marker]), *recent])
        return True
