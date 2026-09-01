"""Desktop bridge protocol regression tests."""

from __future__ import annotations

from cagent.gui.bridge import _jsonable
from cagent.types import Message, RiskLevel, TextPart, ThinkingPart, ToolCallPart


def test_bridge_serializes_int_enum_as_a_risk_name() -> None:
    assert _jsonable(RiskLevel.DANGEROUS) == "dangerous"


def test_bridge_adds_discriminators_to_message_parts() -> None:
    message = Message.assistant(
        ThinkingPart("checking"),
        TextPart("done"),
        ToolCallPart("call-1", "read_file", {"path": "app.py"}),
    )

    encoded = _jsonable(message)

    assert isinstance(encoded, dict)
    assert [part["type"] for part in encoded["parts"]] == [
        "thinking",
        "text",
        "tool_call",
    ]
