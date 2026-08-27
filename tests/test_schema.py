"""Schema reflection and argument validation.

The schema handed to the model is derived from Python annotations, so these
tests are what stop a tool's declared arguments and its actual arguments from
drifting apart. The parsing tests pin down a deliberate asymmetry: lenient where
models are reliably sloppy about JSON types, strict where being wrong would run
the wrong action.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest

from cagent.errors import ToolArgumentError
from cagent.tools.schema import (
    Doc,
    build_function_schema,
    build_object_schema,
    parse_object,
)


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Nested:
    label: Annotated[str, Doc("what to call it")]
    weight: float = 1.0


@dataclass
class Params:
    path: Annotated[str, Doc("file to read")]
    count: Annotated[int, "how many"] = 3
    ratio: float = 0.5
    enabled: bool = False
    mode: Literal["fast", "slow"] = "fast"
    tags: list[str] = field(default_factory=list)
    limit: int | None = None
    colour: Colour = Colour.RED
    nested: Nested | None = None
    many: list[Nested] = field(default_factory=list)
    extras: dict[str, int] = field(default_factory=dict)


class TestBuildObjectSchema:
    def test_required_is_only_fields_without_defaults(self) -> None:
        schema = build_object_schema(Params)
        assert schema["required"] == ["path"]

    def test_object_is_closed(self) -> None:
        # additionalProperties False is what makes a hallucinated argument a
        # validation error rather than a silently ignored one.
        assert build_object_schema(Params)["additionalProperties"] is False

    def test_field_order_is_preserved(self) -> None:
        properties = build_object_schema(Params)["properties"]
        assert list(properties)[:4] == ["path", "count", "ratio", "enabled"]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("path", {"type": "string"}),
            ("count", {"type": "integer"}),
            ("ratio", {"type": "number"}),
            ("enabled", {"type": "boolean"}),
        ],
    )
    def test_primitive_types(self, name: str, expected: dict[str, object]) -> None:
        properties = build_object_schema(Params)["properties"]
        assert properties[name]["type"] == expected["type"]  # type: ignore[index]

    def test_doc_annotation_becomes_description(self) -> None:
        properties = build_object_schema(Params)["properties"]
        assert properties["path"]["description"] == "file to read"  # type: ignore[index]

    def test_bare_string_annotation_is_shorthand_for_doc(self) -> None:
        properties = build_object_schema(Params)["properties"]
        assert properties["count"]["description"] == "how many"  # type: ignore[index]

    def test_literal_becomes_enum_with_inferred_type(self) -> None:
        mode = build_object_schema(Params)["properties"]["mode"]  # type: ignore[index]
        assert mode["enum"] == ["fast", "slow"]
        assert mode["type"] == "string"

    def test_enum_class_uses_values(self) -> None:
        colour = build_object_schema(Params)["properties"]["colour"]  # type: ignore[index]
        assert colour["enum"] == ["red", "blue"]

    def test_list_gets_items(self) -> None:
        tags = build_object_schema(Params)["properties"]["tags"]  # type: ignore[index]
        assert tags["type"] == "array"
        assert tags["items"]["type"] == "string"

    def test_dict_gets_additional_properties_schema(self) -> None:
        extras = build_object_schema(Params)["properties"]["extras"]  # type: ignore[index]
        assert extras["type"] == "object"
        assert extras["additionalProperties"]["type"] == "integer"

    def test_optional_unwraps_to_inner_type_and_is_not_required(self) -> None:
        schema = build_object_schema(Params)
        assert schema["properties"]["limit"]["type"] == "integer"  # type: ignore[index]
        assert "limit" not in schema["required"]  # type: ignore[operator]

    def test_nested_dataclass_recurses(self) -> None:
        nested = build_object_schema(Params)["properties"]["nested"]  # type: ignore[index]
        assert nested["type"] == "object"
        assert nested["properties"]["label"]["description"] == "what to call it"
        assert nested["required"] == ["label"]

    def test_list_of_dataclasses_recurses(self) -> None:
        # multi_edit's `edits` argument depends on this path working.
        many = build_object_schema(Params)["properties"]["many"]  # type: ignore[index]
        assert many["type"] == "array"
        assert many["items"]["properties"]["label"]["type"] == "string"

    def test_unsupported_annotation_fails_loudly(self) -> None:
        @dataclass
        class Bad:
            thing: complex

        # Failing at schema-build time beats shipping a wrong schema to the
        # model and debugging the resulting nonsense arguments.
        with pytest.raises(ToolArgumentError) as caught:
            build_object_schema(Bad)
        assert "thing" in str(caught.value)


class TestParseObject:
    def test_defaults_fill_in(self) -> None:
        parsed = parse_object(Params, {"path": "a.py"})
        assert parsed.count == 3 and parsed.mode == "fast" and parsed.tags == []

    @pytest.mark.parametrize("raw", ["12", 12, 12.0])
    def test_int_accepts_stringly_typed_numbers(self, raw: object) -> None:
        assert parse_object(Params, {"path": "a", "count": raw}).count == 12

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            (1, True),
            ("false", False),
            ("no", False),
            (0, False),
        ],
    )
    def test_bool_accepts_common_spellings(self, raw: object, expected: bool) -> None:
        assert parse_object(Params, {"path": "a", "enabled": raw}).enabled is expected

    def test_int_is_accepted_for_float(self) -> None:
        assert parse_object(Params, {"path": "a", "ratio": 2}).ratio == 2.0

    def test_scalar_is_wrapped_into_a_list(self) -> None:
        # Models routinely send a bare string where an array is declared; the
        # intent is unambiguous, so rejecting it would waste a turn.
        assert parse_object(Params, {"path": "a", "tags": "solo"}).tags == ["solo"]

    def test_enum_accepts_its_value(self) -> None:
        assert parse_object(Params, {"path": "a", "colour": "blue"}).colour is Colour.BLUE

    def test_nested_dataclass_is_constructed(self) -> None:
        parsed = parse_object(Params, {"path": "a", "nested": {"label": "x", "weight": 2}})
        assert parsed.nested is not None
        assert parsed.nested.label == "x" and parsed.nested.weight == 2.0

    def test_list_of_nested_dataclasses_is_constructed(self) -> None:
        parsed = parse_object(Params, {"path": "a", "many": [{"label": "one"}, {"label": "two"}]})
        assert [item.label for item in parsed.many] == ["one", "two"]

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {})
        assert "path" in str(caught.value)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {"path": "a", "nonsense": 1})
        assert "nonsense" in str(caught.value)

    def test_value_outside_literal_is_rejected_and_lists_the_options(self) -> None:
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {"path": "a", "mode": "medium"})
        message = str(caught.value)
        assert "mode" in message and "fast" in message

    def test_uncoercible_int_is_rejected(self) -> None:
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {"path": "a", "count": "not a number"})
        assert "count" in str(caught.value)

    def test_all_field_errors_are_reported_together(self) -> None:
        # One round trip per mistake would be slow and, worse, invites the model
        # to fix one field and break another.
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {"count": "x", "mode": "sideways", "bogus": 1})
        message = str(caught.value)
        for expected in ("path", "count", "mode", "bogus"):
            assert expected in message, message

    def test_error_carries_a_model_actionable_hint(self) -> None:
        with pytest.raises(ToolArgumentError) as caught:
            parse_object(Params, {})
        assert caught.value.as_model_feedback().strip()


class TestBuildFunctionSchema:
    def test_signature_and_docstring_become_schema(self) -> None:
        def fetch(path: str, retries: int = 2, *, verbose: bool = False) -> str:
            """Fetch a thing.

            Args:
                path: where to fetch from
                retries (int): how many times to try
                verbose: whether to say more
                    across a continuation line
            """
            return path

        schema, description = build_function_schema(fetch)
        assert description == "Fetch a thing."
        properties = schema["properties"]
        assert schema["required"] == ["path"]
        assert properties["path"]["description"] == "where to fetch from"  # type: ignore[index]
        assert properties["retries"]["description"] == "how many times to try"  # type: ignore[index]
        assert "continuation line" in properties["verbose"]["description"]  # type: ignore[index]

    def test_self_is_skipped(self) -> None:
        class Holder:
            def method(self, value: str) -> str:
                """Do it.

                Args:
                    value: the value
                """
                return value

        schema, _ = build_function_schema(Holder.method)
        assert list(schema["properties"]) == ["value"]  # type: ignore[arg-type]
