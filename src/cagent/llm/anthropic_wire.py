"""The Anthropic Messages wire adapter.

Speaks the ``/messages`` streaming dialect: named SSE events, content blocks
addressed by index, and strict user/assistant alternation on the way in. As
with every wire module, translation stops here - nothing above this layer sees
a ``content_block_delta`` or a ``tool_use`` block.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import ClassVar

import httpx

from ..errors import (
    AuthError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from ..types import (
    FinishReason,
    Message,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolSpec,
    Usage,
)
from .base import (
    LLMProvider,
    StreamEvent,
    StreamFinished,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallStarted,
    UsageReport,
)
from .retry import RetryPolicy, classify_http_error, with_retries
from .sse import iter_json_events

__all__ = ["AnthropicProvider"]

ANTHROPIC_VERSION = "2023-06-01"
"""API version header value; pinned so behaviour does not shift under us."""

_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "refusal": "content_filter",
}


def _int(value: object) -> int:
    """Read a numeric JSON field defensively; anything else counts as zero."""
    if isinstance(value, int | float):
        return int(value)
    return 0


def _blocks_for(message: Message) -> tuple[str, list[dict[str, object]]]:
    """Map one neutral message onto its wire role and content blocks."""
    if message.role == "assistant":
        blocks: list[dict[str, object]] = []
        for part in message.parts:
            if isinstance(part, ThinkingPart):
                # Unsigned thinking is dropped: the API rejects replayed
                # reasoning it cannot verify.
                if part.signature:
                    blocks.append(
                        {"type": "thinking", "thinking": part.text, "signature": part.signature}
                    )
            elif isinstance(part, TextPart):
                if part.text:
                    blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}
                )
        return "assistant", blocks
    if message.role == "tool":
        return "user", [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
    text = message.text
    return "user", ([{"type": "text", "text": text}] if text else [])


def _wire_messages(messages: Sequence[Message]) -> tuple[list[dict[str, object]], list[str]]:
    """Serialise history into strictly-alternating wire messages.

    System-role turns cannot appear in the messages array, so their text is
    returned separately for the caller to fold into the top-level ``system``
    field. Consecutive same-role wire messages are merged because the API
    enforces strict user/assistant alternation - a tool-result turn followed
    by a user turn must arrive as one user message.
    """
    merged: list[tuple[str, list[dict[str, object]]]] = []
    extra_system: list[str] = []
    for message in messages:
        if message.role == "system":
            if message.text:
                extra_system.append(message.text)
            continue
        role, blocks = _blocks_for(message)
        if not blocks:
            continue
        if merged and merged[-1][0] == role:
            merged[-1][1].extend(blocks)
        else:
            merged.append((role, blocks))
    wire: list[dict[str, object]] = [
        {"role": role, "content": blocks} for role, blocks in merged
    ]
    return wire, extra_system


def _stream_error(payload: dict[str, object]) -> ProviderError:
    """Map an in-stream ``error`` event onto the retry-deciding hierarchy."""
    error = payload.get("error")
    error_type = ""
    message = ""
    if isinstance(error, dict):
        raw_type = error.get("type")
        error_type = raw_type if isinstance(raw_type, str) else ""
        raw_message = error.get("message")
        message = raw_message if isinstance(raw_message, str) else ""
    if not message:
        message = "Provider reported a stream error."
    if error_type == "overloaded_error":
        return TransientProviderError(f"Provider overloaded: {message}")
    if error_type == "rate_limit_error":
        return RateLimitError(message)
    if error_type == "authentication_error":
        return AuthError(message)
    prefix = f"{error_type}: " if error_type else ""
    return ProviderError(f"{prefix}{message}")


class AnthropicProvider(LLMProvider):
    """Adapter for the Anthropic Messages API.

    Wire block indices count every content block (text, thinking, tool_use),
    but the neutral contract numbers tool calls densely from zero, so this
    adapter keeps a wire-index to dense-index map per response. Thinking
    signatures ride on :class:`ThinkingDelta.signature` with empty text, which
    :meth:`LLMProvider.complete` picks up when assembling the message.
    """

    wire: ClassVar[str] = "anthropic"

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec],
    ) -> dict[str, object]:
        wire, extra_system = _wire_messages(messages)
        body: dict[str, object] = {
            "model": self.config.resolved_model,
            "max_tokens": self.config.max_output_tokens,
            "messages": wire,
            "stream": True,
            "temperature": self.config.temperature,
        }
        system_text = "\n\n".join(part for part in (system, *extra_system) if part)
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
                for spec in tools
            ]
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION, "accept": "text/event-stream"}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    def _send(self, payload: dict[str, object]) -> httpx.Response:
        request = self._client.build_request(
            "POST",
            f"{self.config.resolved_base_url}/messages",
            json=payload,
            headers=self._headers(),
        )
        try:
            return self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise TransientProviderError(f"Could not reach the provider: {exc}") from exc

    @staticmethod
    def _drain(response: httpx.Response) -> str:
        """Read and close an error response, tolerating a broken connection."""
        try:
            return response.read().decode("utf-8", "replace")
        except httpx.HTTPError:
            return ""
        finally:
            response.close()

    def _open_stream(self, body: dict[str, object]) -> httpx.Response:
        """POST the request; classify and raise on any non-2xx status."""
        response = self._send(dict(body))
        if response.is_success:
            return response
        status = response.status_code
        headers = dict(response.headers)
        raise classify_http_error(status, self._drain(response), headers)

    @staticmethod
    def _delta_events(
        delta: dict[str, object],
        payload: dict[str, object],
        dense: dict[int, int],
    ) -> Iterator[StreamEvent]:
        """Translate one ``content_block_delta`` into neutral events."""
        kind = delta.get("type")
        if kind == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                yield TextDelta(text)
        elif kind == "thinking_delta":
            thinking = delta.get("thinking")
            if isinstance(thinking, str) and thinking:
                yield ThinkingDelta(thinking)
        elif kind == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str) and partial:
                wire_index = _int(payload.get("index"))
                yield ToolCallArgsDelta(
                    index=dense.setdefault(wire_index, len(dense)),
                    delta=partial,
                )
        elif kind == "signature_delta":
            signature = delta.get("signature")
            if isinstance(signature, str) and signature:
                # complete() reads signatures off ThinkingDelta.signature and
                # appends the (empty) text, so this adds nothing to the trace.
                yield ThinkingDelta(text="", signature=signature)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec] = (),
        abort: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Send one request and yield neutral events as SSE events arrive."""
        body = self._build_body(messages, system=system, tools=tools)
        policy = RetryPolicy(max_retries=self.config.max_retries)
        response = with_retries(lambda: self._open_stream(body), policy, abort=abort)
        self._set_active_response(response)

        input_tokens = 0
        cached_tokens = 0
        output_tokens = 0
        saw_usage = False
        reason: FinishReason | None = None
        dense: dict[int, int] = {}

        def usage_report() -> UsageReport:
            return UsageReport(
                Usage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
            )

        try:
            events = iter_json_events(response.iter_lines())
            while True:
                if abort is not None and abort.is_set():
                    response.close()
                    yield StreamFinished("aborted")
                    return
                try:
                    event_name, payload = next(events)
                except StopIteration:
                    break
                except Exception as exc:  # noqa: BLE001  # cancellation closes the stream
                    # Never reissue mid-stream: replaying half a completion
                    # would corrupt the transcript. The engine decides.
                    if abort is not None and abort.is_set():
                        yield StreamFinished("aborted")
                        return
                    if isinstance(exc, httpx.HTTPError):
                        raise TransientProviderError(f"Stream dropped mid-response: {exc}") from exc
                    raise

                kind = event_name if event_name != "message" else str(payload.get("type", ""))

                if kind == "message_start":
                    started = payload.get("message")
                    if isinstance(started, dict):
                        usage = started.get("usage")
                        if isinstance(usage, dict):
                            input_tokens = _int(usage.get("input_tokens"))
                            cached_tokens = _int(usage.get("cache_read_input_tokens"))
                            saw_usage = True
                elif kind == "content_block_start":
                    block = payload.get("content_block")
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        wire_index = _int(payload.get("index"))
                        raw_id = block.get("id")
                        raw_name = block.get("name")
                        yield ToolCallStarted(
                            index=dense.setdefault(wire_index, len(dense)),
                            id=raw_id if isinstance(raw_id, str) else "",
                            name=raw_name if isinstance(raw_name, str) else "",
                        )
                elif kind == "content_block_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        yield from self._delta_events(delta, payload, dense)
                elif kind == "message_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, dict):
                        stop = delta.get("stop_reason")
                        if isinstance(stop, str) and stop:
                            reason = _STOP_REASONS.get(stop, "stop")
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        output_tokens = _int(usage.get("output_tokens"))
                        saw_usage = True
                elif kind == "message_stop":
                    if saw_usage:
                        yield usage_report()
                    yield StreamFinished(reason if reason is not None else "stop")
                    return
                elif kind == "error":
                    raise _stream_error(payload)
                # ping and unknown events are ignored.
        finally:
            self._clear_active_response(response)
            response.close()

        # Connection closed without message_stop: report what arrived and
        # terminate the stream so the one-StreamFinished contract holds.
        if saw_usage:
            yield usage_report()
        yield StreamFinished(reason if reason is not None else "error")
