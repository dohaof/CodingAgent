"""The transport primitives: SSE parsing, tool-call accumulation, retry, tokens.

These are the pieces that face a provider's real output rather than its
documented output, so the cases worth testing are the malformed ones.
"""

from __future__ import annotations

import threading

import pytest

from cagent.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    ResponseParseError,
    TransientProviderError,
    UserAbort,
)
from cagent.llm.base import (
    StreamFinished,
    TextDelta,
    ToolCallAccumulator,
    ToolCallArgsDelta,
    ToolCallStarted,
    UsageReport,
)
from cagent.llm.retry import RetryPolicy, classify_http_error, with_retries
from cagent.llm.sse import iter_json_events, iter_sse
from cagent.llm.tokens import estimate_message, estimate_messages, estimate_text, estimate_tools
from cagent.types import Message, TextPart, ToolCallPart, ToolSpec, Usage


class TestSSE:
    def test_multiline_data_is_joined_with_newlines(self) -> None:
        events = list(iter_sse(['data: {"a":', "data:  1}", ""]))
        assert len(events) == 1
        assert events[0].data == '{"a":\n 1}'

    def test_event_names_are_captured(self) -> None:
        events = list(iter_sse(["event: message_start", "data: {}", ""]))
        assert events[0].event == "message_start"

    def test_comment_lines_are_ignored(self) -> None:
        events = list(iter_sse([": keep-alive", "data: x", ""]))
        assert [event.data for event in events] == ["x"]

    def test_one_leading_space_after_colon_is_stripped_but_not_two(self) -> None:
        events = list(iter_sse(["data:  padded", ""]))
        assert events[0].data == " padded"

    def test_final_event_without_trailing_blank_line_is_still_emitted(self) -> None:
        # Streams get cut off; dropping the last event would lose a finish_reason.
        events = list(iter_sse(["data: last"]))
        assert [event.data for event in events] == ["last"]

    def test_done_sentinel_is_visible_to_iter_sse(self) -> None:
        events = list(iter_sse(["data: [DONE]", ""]))
        assert events[0].data == "[DONE]"

    def test_iter_json_events_filters_done_and_decodes(self) -> None:
        # An unnamed event is "message", per the SSE default event type.
        decoded = list(iter_json_events(['data: {"x": 1}', "", "data: [DONE]", ""]))
        assert decoded == [("message", {"x": 1})]

    def test_iter_json_events_keeps_event_names(self) -> None:
        decoded = list(iter_json_events(["event: ping", "data: {}", ""]))
        assert decoded == [("ping", {})]

    def test_malformed_json_raises_and_keeps_the_payload(self) -> None:
        # The raw payload rides on the exception rather than in its message, so
        # a log can show what arrived without the message becoming unreadable.
        with pytest.raises(ResponseParseError) as caught:
            list(iter_json_events(["data: {not json", ""]))
        assert "malformed JSON" in str(caught.value)
        assert caught.value.raw is not None and "not json" in caught.value.raw


class TestToolCallAccumulator:
    def test_fragments_are_joined_into_arguments(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c1", name="read_file"))
        for piece in ('{"pa', 'th": "a', '.py"}'):
            acc.feed(ToolCallArgsDelta(index=0, delta=piece))
        (call,) = acc.finish()
        assert call.name == "read_file" and call.arguments == {"path": "a.py"}

    def test_name_split_across_events_is_concatenated(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c1", name="read_"))
        acc.feed(ToolCallStarted(index=0, id="", name="file"))
        assert acc.finish()[0].name == "read_file"

    def test_parallel_calls_are_kept_apart_by_index(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="a", name="one"))
        acc.feed(ToolCallStarted(index=1, id="b", name="two"))
        acc.feed(ToolCallArgsDelta(index=1, delta='{"x": 2}'))
        acc.feed(ToolCallArgsDelta(index=0, delta='{"x": 1}'))
        calls = acc.finish()
        assert [c.name for c in calls] == ["one", "two"]
        assert [c.arguments["x"] for c in calls] == [1, 2]

    def test_empty_arguments_become_an_empty_object(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c", name="list_dir"))
        acc.feed(ToolCallArgsDelta(index=0, delta=""))
        assert acc.finish()[0].arguments == {}

    def test_malformed_json_is_preserved_rather_than_raised(self) -> None:
        # Losing the turn to an exception would be worse than telling the model
        # its JSON was broken, which it can fix.
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c", name="grep"))
        acc.feed(ToolCallArgsDelta(index=0, delta='{"query": broken'))
        (call,) = acc.finish()
        assert call.arguments == {}
        assert "broken" in call.raw_arguments
        assert acc.malformed == [0]

    def test_valid_json_of_the_wrong_shape_counts_as_malformed(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c", name="grep"))
        acc.feed(ToolCallArgsDelta(index=0, delta='["not", "an object"]'))
        assert acc.malformed == [] and acc.finish()[0].arguments == {}
        assert acc.malformed == [0]

    def test_missing_id_gets_a_synthetic_one(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=3, id="", name="x"))
        assert acc.finish()[0].id == "call_3"

    def test_arguments_arriving_before_a_start_event_are_kept(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallArgsDelta(index=0, delta='{"a": 1}'))
        assert acc.finish()[0].arguments == {"a": 1}

    def test_finish_is_idempotent(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c", name="x"))
        acc.feed(ToolCallArgsDelta(index=0, delta="oops"))
        first, second = acc.finish(), acc.finish()
        assert first == second and acc.malformed == [0]

    def test_reset_clears_state(self) -> None:
        acc = ToolCallAccumulator()
        acc.feed(ToolCallStarted(index=0, id="c", name="x"))
        acc.reset()
        assert acc.finish() == [] and acc.malformed == []


class TestRetry:
    def test_returns_the_first_success_without_sleeping(self, no_sleep) -> None:
        result = with_retries(lambda: "ok", RetryPolicy(max_retries=3), sleep=no_sleep)
        assert result == "ok" and no_sleep.calls == []

    def test_retries_a_transient_error_then_succeeds(self, no_sleep) -> None:
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TransientProviderError("upstream hiccup")
            return "ok"

        assert with_retries(flaky, RetryPolicy(max_retries=5), sleep=no_sleep) == "ok"
        assert attempts["n"] == 3 and len(no_sleep.calls) == 2

    def test_auth_error_is_not_retried(self, no_sleep) -> None:
        # A bad key will still be bad in 600ms; retrying only delays the message.
        attempts = {"n": 0}

        def failing() -> str:
            attempts["n"] += 1
            raise AuthError("bad key")

        with pytest.raises(AuthError):
            with_retries(failing, RetryPolicy(max_retries=5), sleep=no_sleep)
        assert attempts["n"] == 1 and no_sleep.calls == []

    def test_exhaustion_reraises_the_last_error(self, no_sleep) -> None:
        def always() -> str:
            raise RateLimitError("slow down")

        with pytest.raises(RateLimitError):
            with_retries(always, RetryPolicy(max_retries=2), sleep=no_sleep)
        assert len(no_sleep.calls) == 2

    def test_backoff_grows(self) -> None:
        policy = RetryPolicy(max_retries=5, base_delay=1.0, max_delay=60.0, jitter=0.0)
        delays = [policy.delay_for(attempt) for attempt in range(4)]
        assert delays == sorted(delays) and delays[-1] > delays[0]

    def test_backoff_is_capped(self) -> None:
        policy = RetryPolicy(max_retries=20, base_delay=1.0, max_delay=5.0, jitter=0.0)
        assert policy.delay_for(15) <= 5.0

    def test_server_retry_after_wins_when_longer(self) -> None:
        policy = RetryPolicy(base_delay=0.5, max_delay=60.0, jitter=0.0)
        assert policy.delay_for(0, retry_after=30.0) >= 30.0

    def test_abort_stops_retrying(self, no_sleep) -> None:
        # Reported as UserAbort, not as the provider error: the run ended because
        # the user said so, and the distinction matters to the caller.
        abort = threading.Event()
        attempts = {"n": 0}

        def failing() -> str:
            attempts["n"] += 1
            abort.set()
            raise TransientProviderError("later")

        with pytest.raises(UserAbort):
            with_retries(failing, RetryPolicy(max_retries=5), sleep=no_sleep, abort=abort)
        assert attempts["n"] == 1

    def test_on_retry_is_notified(self, no_sleep) -> None:
        seen: list[int] = []
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TransientProviderError("x")
            return "ok"

        with_retries(
            flaky,
            RetryPolicy(max_retries=3),
            on_retry=lambda attempt, exc, delay: seen.append(attempt),
            sleep=no_sleep,
        )
        assert seen == [1]

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, AuthError),
            (403, AuthError),
            (429, RateLimitError),
            (408, TransientProviderError),
            (500, TransientProviderError),
            (502, TransientProviderError),
            (503, TransientProviderError),
            (404, ProviderError),
        ],
    )
    def test_status_classification(self, status: int, expected: type[Exception]) -> None:
        assert isinstance(classify_http_error(status, "{}"), expected)

    def test_context_length_400_is_distinguished(self) -> None:
        # Worth its own class: the fix is compaction, not a retry.
        error = classify_http_error(
            400, '{"error": {"message": "maximum context length exceeded"}}'
        )
        assert isinstance(error, ContextOverflowError)

    def test_retry_after_header_is_parsed(self) -> None:
        error = classify_http_error(429, "{}", {"retry-after": "7"})
        assert isinstance(error, RateLimitError) and error.retry_after == 7.0


class TestTokenEstimation:
    def test_cjk_costs_more_per_character_than_ascii(self) -> None:
        # A per-character heuristic tuned on English badly underestimates
        # Chinese, which matters when the budget decides on compaction.
        assert estimate_text("中文字符测试内容") > estimate_text("abcdefgh")

    def test_longer_text_costs_more(self) -> None:
        assert estimate_text("hello world " * 50) > estimate_text("hello world")

    def test_empty_text_is_free(self) -> None:
        assert estimate_text("") == 0

    def test_message_includes_structural_overhead(self) -> None:
        assert estimate_message(Message.user("hi")) > estimate_text("hi")

    def test_tool_call_arguments_are_counted(self) -> None:
        bare = Message.assistant(TextPart(""))
        with_call = Message.assistant(
            ToolCallPart(id="c", name="edit_file", arguments={"path": "x" * 200})
        )
        assert estimate_message(with_call) > estimate_message(bare)

    def test_cached_estimate_is_reused(self) -> None:
        message = Message.user("some text")
        first = estimate_message(message)
        assert message.token_estimate == first
        message.token_estimate = 999
        assert estimate_message(message) == 999

    def test_tools_and_system_are_included_in_a_request(self) -> None:
        spec = ToolSpec(name="t", description="d" * 100, input_schema={"type": "object"})
        with_extras = estimate_messages([Message.user("x")], system="s" * 200, tools=[spec])
        assert with_extras > estimate_messages([Message.user("x")])

    def test_tools_estimate_scales_with_count(self) -> None:
        one = ToolSpec(name="a", description="d", input_schema={"type": "object"})
        assert estimate_tools([one, one]) > estimate_tools([one])

    def test_heuristic_is_used_when_tiktoken_is_absent(self, monkeypatch) -> None:
        import cagent.llm.tokens as tokens

        monkeypatch.setattr(tokens, "_encoder", lambda model: None)
        tokens._encoder.cache_clear() if hasattr(tokens._encoder, "cache_clear") else None
        assert estimate_text("hello world", model="whatever") > 0


class TestStreamAssembly:
    def test_complete_assembles_parts_and_usage(self, config, scripted) -> None:
        provider = scripted(
            config,
            [
                [
                    TextDelta("Hello "),
                    TextDelta("world"),
                    ToolCallStarted(index=0, id="c1", name="read_file"),
                    ToolCallArgsDelta(index=0, delta='{"path": "a.py"}'),
                    UsageReport(Usage(prompt_tokens=10, completion_tokens=4)),
                    StreamFinished("tool_calls"),
                ]
            ],
        )
        result = provider.complete([Message.user("hi")], system="sys")
        assert result.message.text == "Hello world"
        assert [c.name for c in result.message.tool_calls] == ["read_file"]
        assert result.finish_reason == "tool_calls"
        assert result.usage.prompt_tokens == 10
        assert result.latency_s >= 0

    def test_on_event_sees_every_event(self, config, scripted) -> None:
        provider = scripted(config, [[TextDelta("a"), TextDelta("b"), StreamFinished("stop")]])
        seen: list[object] = []
        provider.complete([Message.user("hi")], system="", on_event=seen.append)
        assert len(seen) == 3

    def test_missing_finish_reason_is_inferred_from_content(self, config, scripted) -> None:
        provider = scripted(config, [[TextDelta("answer")]])
        result = provider.complete([Message.user("hi")], system="")
        assert result.finish_reason == "stop" and result.message.text == "answer"
