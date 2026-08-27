"""The OpenAI Chat Completions wire adapter.

Speaks the ``/chat/completions`` streaming dialect that OpenAI defined and that
DeepSeek, Moonshot, DashScope, OpenRouter, and Ollama emulate. Translation
happens at this boundary only: requests are built from neutral
:class:`~cagent.types.Message` parts and responses are re-emitted as
:data:`~cagent.llm.base.StreamEvent` values, so no ``choices[].delta`` shape
escapes this module.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from typing import ClassVar

import httpx

from ..config import AgentConfig
from ..errors import TransientProviderError
from ..types import FinishReason, Message, ToolCallPart, ToolSpec, Usage
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

__all__ = ["OpenAIProvider"]

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


def _int(value: object) -> int:
    """Read a numeric JSON field defensively; anything else counts as zero."""
    if isinstance(value, int | float):
        return int(value)
    return 0


def _wire_tool_call(part: ToolCallPart) -> dict[str, object]:
    """Serialise one already-made tool call for replay in the history."""
    arguments = json.dumps(part.arguments) if part.arguments else (part.raw_arguments or "{}")
    return {
        "id": part.id,
        "type": "function",
        "function": {"name": part.name, "arguments": arguments},
    }


def _parse_usage(payload: dict[str, object]) -> Usage:
    """Fold a wire ``usage`` object into the neutral counters."""
    prompt_details = payload.get("prompt_tokens_details")
    cached = _int(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else 0
    completion_details = payload.get("completion_tokens_details")
    reasoning = (
        _int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else 0
    )
    return Usage(
        prompt_tokens=_int(payload.get("prompt_tokens")),
        completion_tokens=_int(payload.get("completion_tokens")),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


class OpenAIProvider(LLMProvider):
    """Adapter for OpenAI-compatible chat-completions endpoints.

    Quirks this adapter absorbs so callers never see them:

    * ``stream_options`` is sent for usage accounting, but a strict gateway
      that 400s on the unknown field gets the request reissued once without
      it (and the field is omitted for the rest of the session).
    * Reasoning deltas arrive as ``reasoning_content`` (DeepSeek) or
      ``reasoning`` (OpenRouter); both become :class:`ThinkingDelta`.
    * Replayed assistant turns drop :class:`~cagent.types.ThinkingPart`
      entirely, because reasoning APIs reject their own traces echoed back.
    """

    wire: ClassVar[str] = "openai"

    def __init__(self, config: AgentConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config, client=client)
        self._omit_stream_options = False

    @staticmethod
    def _wire_messages(messages: Sequence[Message], system: str) -> list[dict[str, object]]:
        """Serialise the neutral history into chat-completions messages."""
        wire: list[dict[str, object]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for message in messages:
            if message.role == "tool":
                # One wire message per result: the API keys each result to its
                # originating call id individually.
                wire.extend(
                    {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
                    for result in message.tool_results
                )
            elif message.role == "assistant":
                # message.text skips ThinkingParts, dropping replayed reasoning.
                entry: dict[str, object] = {"role": "assistant", "content": message.text or None}
                calls = [_wire_tool_call(part) for part in message.tool_calls]
                if calls:
                    entry["tool_calls"] = calls
                wire.append(entry)
            else:
                wire.append({"role": message.role, "content": message.text})
        return wire

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec],
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.config.resolved_model,
            "messages": self._wire_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                }
                for spec in tools
            ]
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "text/event-stream"}
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _send(self, payload: dict[str, object]) -> httpx.Response:
        request = self._client.build_request(
            "POST",
            f"{self.config.resolved_base_url}/chat/completions",
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
        """POST the request and return the open streaming response.

        A 400 whose body mentions ``stream_options`` is reissued once without
        that field, because some OpenAI-compatible gateways reject fields they
        do not know. Any other non-2xx becomes the classified provider error.
        """
        payload = dict(body)
        if self._omit_stream_options:
            payload.pop("stream_options", None)

        response = self._send(payload)
        if response.is_success:
            return response

        status = response.status_code
        headers = dict(response.headers)
        text = self._drain(response)

        if status == 400 and "stream_options" in payload and "stream_options" in text:
            self._omit_stream_options = True
            payload.pop("stream_options")
            response = self._send(payload)
            if response.is_success:
                return response
            status = response.status_code
            headers = dict(response.headers)
            text = self._drain(response)

        raise classify_http_error(status, text, headers)

    @staticmethod
    def _delta_events(delta: dict[str, object], seen: set[int]) -> Iterator[StreamEvent]:
        """Translate one ``choices[].delta`` object into neutral events."""
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield TextDelta(content)

        reasoning = delta.get("reasoning_content")
        if not (isinstance(reasoning, str) and reasoning):
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            yield ThinkingDelta(reasoning)

        fragments = delta.get("tool_calls")
        if not isinstance(fragments, list):
            return
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            raw_index = fragment.get("index")
            index = raw_index if isinstance(raw_index, int) else 0
            raw_id = fragment.get("id")
            call_id = raw_id if isinstance(raw_id, str) else ""
            name = ""
            arguments = ""
            function = fragment.get("function")
            if isinstance(function, dict):
                raw_name = function.get("name")
                name = raw_name if isinstance(raw_name, str) else ""
                raw_arguments = function.get("arguments")
                arguments = raw_arguments if isinstance(raw_arguments, str) else ""
            if index not in seen:
                seen.add(index)
                yield ToolCallStarted(index=index, id=call_id, name=name)
            elif call_id or name:
                # The accumulator concatenates name fragments and overwrites
                # ids, so later id/name text is re-emitted exactly as sent.
                yield ToolCallStarted(index=index, id=call_id, name=name)
            if arguments:
                yield ToolCallArgsDelta(index=index, delta=arguments)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolSpec] = (),
        abort: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Send one request and yield neutral events as chunks arrive."""
        body = self._build_body(messages, system=system, tools=tools)
        policy = RetryPolicy(max_retries=self.config.max_retries)
        response = with_retries(lambda: self._open_stream(body), policy, abort=abort)

        finish: FinishReason | None = None
        usage: Usage | None = None
        seen: set[int] = set()

        try:
            events = iter_json_events(response.iter_lines())
            while True:
                if abort is not None and abort.is_set():
                    response.close()
                    yield StreamFinished("aborted")
                    return
                try:
                    _, payload = next(events)
                except StopIteration:
                    break
                except httpx.HTTPError as exc:
                    # Never reissue mid-stream: replaying half a completion
                    # would corrupt the transcript. The engine decides.
                    raise TransientProviderError(f"Stream dropped mid-response: {exc}") from exc

                choices = payload.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            yield from self._delta_events(delta, seen)
                        raw_reason = choice.get("finish_reason")
                        if isinstance(raw_reason, str) and raw_reason:
                            finish = _FINISH_REASONS.get(raw_reason, "stop")

                usage_payload = payload.get("usage")
                if isinstance(usage_payload, dict):
                    usage = _parse_usage(usage_payload)
        finally:
            response.close()

        if usage is not None:
            yield UsageReport(usage)
        yield StreamFinished(finish if finish is not None else "stop")
