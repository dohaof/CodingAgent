"""The wire adapters, driven by a mocked transport.

Two things are worth pinning down. First, the request body: a provider that
serialises history wrongly fails at the far end with a 400 whose message rarely
names the real problem, so the shape is asserted here. Second, the translation
of a real streamed response into the neutral event model.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from cagent.config import AgentConfig
from cagent.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from cagent.llm.anthropic_wire import AnthropicProvider
from cagent.llm.factory import WIRE_IMPLEMENTATIONS, build_provider
from cagent.llm.openai_wire import OpenAIProvider
from cagent.types import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)
from tests.conftest import anthropic_sse, sse_body

TOOLS = (
    ToolSpec(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
)

# A history with the shape that breaks naive serialisers: an assistant turn
# carrying both thinking and a tool call, then that call's result.
HISTORY = [
    Message.user("fix the bug"),
    Message.assistant(
        ThinkingPart("I should look at the file", signature="sig-1"),
        TextPart("Reading it now."),
        ToolCallPart(id="call_1", name="read_file", arguments={"path": "a.py"}),
    ),
    Message.from_tool_results([ToolResultPart(call_id="call_1", content="1\tdef f(): ...")]),
    Message.user("thanks, continue"),
]


def transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def capture(body: bytes, *, status: int = 200) -> tuple[httpx.Client, list[dict]]:
    """A client that returns ``body`` and records every request body sent."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            status,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    return transport(handler), seen


def openai_stream() -> bytes:
    """A realistic OpenAI stream: prose, a split tool call, then usage."""
    return sse_body(
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Look"}}]},
            {"choices": [{"index": 0, "delta": {"content": "ing."}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {"name": "read_file", "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 18,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
        ]
    )


class TestOpenAIWire:
    def test_stream_assembles_text_tool_call_and_usage(self, config: AgentConfig) -> None:
        client, _ = capture(openai_stream())
        provider = OpenAIProvider(config, client=client)
        result = provider.complete([Message.user("hi")], system="sys", tools=TOOLS)

        assert result.message.text == "Looking."
        (call,) = result.message.tool_calls
        assert call.name == "read_file" and call.arguments == {"path": "a.py"}
        assert result.finish_reason == "tool_calls"
        assert result.usage.prompt_tokens == 120
        assert result.usage.cached_tokens == 40
        assert result.usage.reasoning_tokens == 5

    def test_reasoning_content_becomes_thinking(self, config: AgentConfig) -> None:
        client, _ = capture(openai_stream())
        provider = OpenAIProvider(config, client=client)
        result = provider.complete([Message.user("hi")], system="sys")
        thinking = [p for p in result.message.parts if isinstance(p, ThinkingPart)]
        assert thinking and thinking[0].text == "hmm"

    def test_system_prompt_is_the_first_message(self, config: AgentConfig) -> None:
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete([Message.user("hi")], system="be careful")
        assert seen[0]["messages"][0] == {"role": "system", "content": "be careful"}

    def test_tools_are_declared_in_function_form(self, config: AgentConfig) -> None:
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete(
            [Message.user("hi")], system="", tools=TOOLS
        )
        (declared,) = seen[0]["tools"]
        assert declared["type"] == "function"
        assert declared["function"]["name"] == "read_file"
        assert declared["function"]["parameters"]["required"] == ["path"]

    def test_tools_key_is_omitted_when_there_are_none(self, config: AgentConfig) -> None:
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete([Message.user("hi")], system="")
        assert "tools" not in seen[0]

    def test_assistant_tool_call_is_serialised_with_json_arguments(
        self, config: AgentConfig
    ) -> None:
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete(HISTORY, system="")
        assistant = next(m for m in seen[0]["messages"] if m["role"] == "assistant")
        (call,) = assistant["tool_calls"]
        assert call["id"] == "call_1"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}

    def test_thinking_is_not_replayed(self, config: AgentConfig) -> None:
        # Reasoning APIs reject their own replayed reasoning, so it is dropped.
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete(HISTORY, system="")
        assert "I should look at the file" not in json.dumps(seen[0])

    def test_each_tool_result_becomes_its_own_wire_message(self, config: AgentConfig) -> None:
        history = [
            Message.assistant(
                ToolCallPart(id="a", name="read_file", arguments={}),
                ToolCallPart(id="b", name="read_file", arguments={}),
            ),
            Message.from_tool_results(
                [ToolResultPart("a", "first"), ToolResultPart("b", "second")]
            ),
        ]
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete(history, system="")
        tool_messages = [m for m in seen[0]["messages"] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
        assert [m["content"] for m in tool_messages] == ["first", "second"]

    def test_streaming_options_request_usage(self, config: AgentConfig) -> None:
        client, seen = capture(openai_stream())
        OpenAIProvider(config, client=client).complete([Message.user("hi")], system="")
        assert seen[0]["stream"] is True
        assert seen[0]["stream_options"] == {"include_usage": True}

    def test_cancelled_owned_client_is_recreated_for_the_next_turn(
        self, config: AgentConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clients: list[httpx.Client] = []
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=openai_stream(),
                headers={"content-type": "text/event-stream"},
            )

        clients.extend((transport(handler), transport(handler)))
        monkeypatch.setattr(
            OpenAIProvider,
            "_build_client",
            staticmethod(lambda _config: clients.pop(0)),
        )
        provider = OpenAIProvider(config)
        first_client = provider.client

        # This is the race where Ctrl+C arrives after a turn has stopped
        # streaming but before the next input is accepted.
        provider.cancel()
        assert first_client.is_closed

        result = provider.complete([Message.user("continue")], system="")

        assert result.message.text == "Looking."
        assert len(seen) == 1

    def test_bearer_header_is_sent(self, config: AgentConfig) -> None:
        headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers)
            return httpx.Response(200, content=openai_stream())

        provider = OpenAIProvider(config, client=transport(handler))
        provider.complete([Message.user("hi")], system="")
        assert headers[0]["authorization"] == "Bearer test-key"

    def test_bearer_header_is_omitted_without_a_key(self, tmp_path) -> None:
        # A local Ollama has no key, and sending "Bearer None" would be worse
        # than sending nothing.
        headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers)
            return httpx.Response(200, content=openai_stream())

        config = AgentConfig(
            workspace=tmp_path,
            base_url="http://localhost:11434/v1",
            model="local-model",
            api_key=None,
            requires_key=False,
        )
        provider = OpenAIProvider(config, client=transport(handler))
        provider.complete([Message.user("hi")], system="")
        assert "authorization" not in headers[0]

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(401, AuthError), (429, RateLimitError), (503, TransientProviderError)],
    )
    def test_http_errors_are_classified(
        self, config: AgentConfig, status: int, expected: type[Exception]
    ) -> None:
        config.max_retries = 0

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "nope"}})

        provider = OpenAIProvider(config, client=transport(handler))
        with pytest.raises(expected):
            provider.complete([Message.user("hi")], system="")

    def test_midstream_transport_failure_is_transient_not_retried(
        self, config: AgentConfig
    ) -> None:
        # Replaying half a completion would corrupt the transcript, so the
        # adapter surfaces the error and lets the engine decide.
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1

            def broken() -> object:
                yield b'data: {"choices": [{"index": 0, "delta": {"content": "par'
                raise httpx.ReadError("connection lost")

            return httpx.Response(200, content=broken())

        provider = OpenAIProvider(config, client=transport(handler))
        with pytest.raises(TransientProviderError):
            provider.complete([Message.user("hi")], system="")
        assert calls["n"] == 1

    def test_gateway_rejecting_stream_options_is_retried_without_them(
        self, config: AgentConfig
    ) -> None:
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "stream_options" in body:
                return httpx.Response(
                    400, json={"error": {"message": "unknown parameter stream_options"}}
                )
            return httpx.Response(200, content=openai_stream())

        provider = OpenAIProvider(config, client=transport(handler))
        result = provider.complete([Message.user("hi")], system="")
        assert result.message.text == "Looking."
        assert len(bodies) == 2 and "stream_options" not in bodies[1]


def anthropic_stream() -> bytes:
    """A realistic Anthropic stream: text, then a tool_use block, then usage."""
    return anthropic_sse(
        [
            (
                "message_start",
                {
                    "message": {
                        "usage": {"input_tokens": 90, "cache_read_input_tokens": 30},
                    }
                },
            ),
            ("ping", {}),
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "text_delta", "text": "Checking."}},
            ),
            ("content_block_stop", {"index": 0}),
            (
                "content_block_start",
                {
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file"},
                },
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path"'}},
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": ': "a.py"}'}},
            ),
            ("content_block_stop", {"index": 1}),
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 22}},
            ),
            ("message_stop", {}),
        ]
    )


class TestAnthropicWire:
    def test_stream_assembles_text_tool_call_and_usage(self, config: AgentConfig) -> None:
        client, _ = capture(anthropic_stream())
        provider = AnthropicProvider(config, client=client)
        result = provider.complete([Message.user("hi")], system="sys", tools=TOOLS)

        assert result.message.text == "Checking."
        (call,) = result.message.tool_calls
        assert call.name == "read_file" and call.arguments == {"path": "a.py"}
        assert result.finish_reason == "tool_calls"
        assert result.usage.prompt_tokens == 90
        assert result.usage.cached_tokens == 30
        assert result.usage.completion_tokens == 22

    def test_system_goes_top_level_not_into_messages(self, config: AgentConfig) -> None:
        client, seen = capture(anthropic_stream())
        AnthropicProvider(config, client=client).complete([Message.user("hi")], system="rules")
        assert seen[0]["system"] == "rules"
        assert all(m["role"] != "system" for m in seen[0]["messages"])

    def test_max_tokens_is_always_sent(self, config: AgentConfig) -> None:
        # The Messages API rejects a request without it.
        client, seen = capture(anthropic_stream())
        AnthropicProvider(config, client=client).complete([Message.user("hi")], system="")
        assert seen[0]["max_tokens"] == config.max_output_tokens

    def test_tool_results_become_user_tool_result_blocks(self, config: AgentConfig) -> None:
        client, seen = capture(anthropic_stream())
        AnthropicProvider(config, client=client).complete(HISTORY, system="")
        blocks = [
            block
            for message in seen[0]["messages"]
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        assert blocks and blocks[0]["tool_use_id"] == "call_1"

    def test_consecutive_same_role_messages_are_merged(self, config: AgentConfig) -> None:
        # The API requires strict alternation, and a tool result is a user turn,
        # so it would otherwise sit next to the following real user turn.
        client, seen = capture(anthropic_stream())
        AnthropicProvider(config, client=client).complete(HISTORY, system="")
        roles = [m["role"] for m in seen[0]["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles

    def test_signed_thinking_is_replayed_and_unsigned_is_dropped(
        self, config: AgentConfig
    ) -> None:
        client, seen = capture(anthropic_stream())
        history = [
            Message.assistant(ThinkingPart("kept", signature="sig"), TextPart("a")),
            Message.user("next"),
            Message.assistant(ThinkingPart("dropped"), TextPart("b")),
            Message.user("again"),
        ]
        AnthropicProvider(config, client=client).complete(history, system="")
        body = json.dumps(seen[0])
        assert "kept" in body and "dropped" not in body

    def test_tools_use_input_schema(self, config: AgentConfig) -> None:
        client, seen = capture(anthropic_stream())
        AnthropicProvider(config, client=client).complete(
            [Message.user("hi")], system="", tools=TOOLS
        )
        (declared,) = seen[0]["tools"]
        assert declared["name"] == "read_file" and "input_schema" in declared

    def test_api_key_header_is_used(self, config: AgentConfig) -> None:
        headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers)
            return httpx.Response(200, content=anthropic_stream())

        AnthropicProvider(config, client=transport(handler)).complete(
            [Message.user("hi")], system=""
        )
        assert headers[0]["x-api-key"] == "test-key"
        assert headers[0]["anthropic-version"] == "2023-06-01"

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            ("overloaded_error", TransientProviderError),
            ("rate_limit_error", RateLimitError),
            ("authentication_error", AuthError),
            ("something_else", ProviderError),
        ],
    )
    def test_error_events_are_mapped(
        self, config: AgentConfig, error_type: str, expected: type[Exception]
    ) -> None:
        config.max_retries = 0
        body = anthropic_sse(
            [("error", {"error": {"type": error_type, "message": "upstream said no"}})]
        )
        client, _ = capture(body)
        with pytest.raises(expected):
            AnthropicProvider(config, client=client).complete([Message.user("hi")], system="")

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
        ],
    )
    def test_stop_reason_mapping(
        self, config: AgentConfig, stop_reason: str, expected: str
    ) -> None:
        body = anthropic_sse(
            [
                ("message_start", {"message": {"usage": {"input_tokens": 1}}}),
                ("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {}}),
                ("message_stop", {}),
            ]
        )
        client, _ = capture(body)
        result = AnthropicProvider(config, client=client).complete(
            [Message.user("hi")], system=""
        )
        assert result.finish_reason == expected


def endpoint_config(tmp_path, **overrides: object) -> AgentConfig:
    """A minimal complete endpoint: url, model, key, and a wire."""
    settings: dict[str, object] = {
        "workspace": tmp_path,
        "base_url": "https://x.invalid/v1",
        "model": "m",
        "api_key": "k",
    }
    settings.update(overrides)
    return AgentConfig(**settings)  # type: ignore[arg-type]


class TestFactory:
    def test_both_wires_are_registered(self) -> None:
        assert set(WIRE_IMPLEMENTATIONS) == {"openai", "anthropic"}

    def test_the_default_wire_is_openai(self, tmp_path) -> None:
        # Almost every endpoint emulates Chat Completions, so it is the default
        # and the Anthropic shape is the thing you opt into.
        assert isinstance(build_provider(endpoint_config(tmp_path)), OpenAIProvider)

    def test_the_anthropic_wire_can_be_selected(self, tmp_path) -> None:
        config = endpoint_config(tmp_path, wire="anthropic")
        assert isinstance(build_provider(config), AnthropicProvider)

    def test_an_unknown_wire_is_a_config_error(self, tmp_path) -> None:
        config = endpoint_config(tmp_path)
        config.wire = "carrier-pigeon"  # type: ignore[assignment]
        with pytest.raises(ConfigError):
            build_provider(config)
