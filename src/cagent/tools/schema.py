"""Reflection from Python type annotations to JSON Schema, and back.

A tool declares its arguments once, as an annotated dataclass or an annotated
function signature. :func:`build_object_schema` derives the JSON Schema the
provider is shown, and :func:`parse_object` validates the model's JSON back into
that same dataclass. The two directions read the same annotations, so a schema
cannot drift from the code that consumes it.

Two asymmetric policies are deliberate. Schema construction fails loudly, at
import time, on an annotation it cannot express: shipping a silently wrong
schema to the model is far worse than not starting. Argument parsing is lenient
about the things models get reliably wrong (a stringified int, a bare scalar
where a list belongs) and strict about everything else, and every rejection
carries a hint written as an instruction, because that text goes back into the
transcript as the model's next observation.

Standard library plus :mod:`cagent.errors` only.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass
from typing import (
    Annotated,
    Any,
    Literal,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ..errors import ToolArgumentError

__all__ = [
    "CONTEXT_TYPE_NAME",
    "Doc",
    "build_function_schema",
    "build_object_schema",
    "is_context_annotation",
    "parse_docstring",
    "parse_object",
]

T = TypeVar("T")

CONTEXT_TYPE_NAME = "ToolContext"
"""Annotations with this type name are host-injected, not model-supplied.

Matched by name rather than by identity so this module stays importable by
:mod:`cagent.tools.base`, which defines ``ToolContext`` and imports from here.
"""

_MAX_SHOWN = 120
"""Received values are truncated to this many characters in error messages."""

_PRIMITIVES: dict[object, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True, slots=True)
class Doc:
    """Field documentation carried in :data:`~typing.Annotated` metadata.

    Used as ``Annotated[int, Doc("1-based line number")]``. A bare string in the
    metadata is accepted as shorthand for ``Doc(...)``.
    """

    text: str


def _truncate(value: object, limit: int = _MAX_SHOWN) -> str:
    """Render ``value`` for an error message, bounded in length."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars)"


def _unwrap_annotated(hint: object) -> tuple[object, str | None]:
    """Split ``Annotated[T, ...]`` into ``T`` and its description, if any.

    The last ``Doc`` or plain string in the metadata wins, so a nested
    annotation can be re-documented by the outer one.
    """
    if get_origin(hint) is not Annotated:
        return hint, None
    args = get_args(hint)
    inner = args[0]
    description: str | None = None
    for meta in args[1:]:
        if isinstance(meta, Doc):
            description = meta.text
        elif isinstance(meta, str):
            description = meta
    nested, nested_description = _unwrap_annotated(inner)
    return nested, description if description is not None else nested_description


def _split_optional(hint: object) -> tuple[object, bool]:
    """Split ``T | None`` into ``(T, True)``; anything else is ``(hint, False)``.

    A union of several non-``None`` members is not representable in the subset of
    JSON Schema this module emits, so it is left for the caller to reject.
    """
    origin = get_origin(hint)
    if origin is not types.UnionType and origin is not Union:
        return hint, False
    args = get_args(hint)
    concrete = [arg for arg in args if arg is not type(None)]
    if len(concrete) == len(args):
        return hint, False
    if len(concrete) == 1:
        return concrete[0], True
    return Union[tuple(concrete)], True  # noqa: UP007  # re-formed union, rejected downstream


def _is_dataclass_type(hint: object) -> TypeGuard[type]:
    return isinstance(hint, type) and dataclasses.is_dataclass(hint)


def _is_enum_type(hint: object) -> TypeGuard[type[enum.Enum]]:
    return isinstance(hint, type) and issubclass(hint, enum.Enum)


def _unsupported(owner: str, field_name: str, hint: object) -> ToolArgumentError:
    """The single failure mode of schema construction."""
    return ToolArgumentError(
        f"Cannot build a JSON Schema for {owner}.{field_name}: "
        f"unsupported annotation {hint!r}.",
        "Annotate this parameter with str, int, float, bool, a Literal, an Enum, "
        "a list[T], a dict[str, T], a nested dataclass, or an optional of those.",
    )


def _literal_schema(owner: str, field_name: str, hint: object) -> dict[str, object]:
    """Emit ``{"type": ..., "enum": [...]}`` for a ``Literal``."""
    choices = get_args(hint)
    kinds = {type(choice) for choice in choices}
    if len(kinds) != 1:
        raise ToolArgumentError(
            f"Literal for {owner}.{field_name} mixes value types "
            f"({', '.join(sorted(kind.__name__ for kind in kinds))}).",
            "Use a Literal whose members are all the same type, or split the "
            "parameter into two.",
        )
        # A mixed Literal has no single JSON type, and omitting "type" would let
        # the provider validate loosely against a schema we cannot enforce.
    kind = kinds.pop()
    json_type = _PRIMITIVES.get(kind)
    if json_type is None:
        raise _unsupported(owner, field_name, hint)
    return {"type": json_type, "enum": list(choices)}


def _enum_schema(owner: str, field_name: str, hint: type[enum.Enum]) -> dict[str, object]:
    """Emit an enum schema over the members' ``.value``."""
    values = [member.value for member in hint]
    kinds = {type(value) for value in values}
    if len(kinds) != 1:
        raise ToolArgumentError(
            f"Enum {hint.__name__} for {owner}.{field_name} mixes value types.",
            "Give every member of this enum a value of the same primitive type.",
        )
    json_type = _PRIMITIVES.get(kinds.pop())
    if json_type is None:
        raise _unsupported(owner, field_name, hint)
    return {"type": json_type, "enum": values}


def _type_schema(owner: str, field_name: str, hint: object) -> dict[str, object]:
    """Schema for one annotation, ignoring optionality and descriptions."""
    hint, _ = _unwrap_annotated(hint)

    if hint in _PRIMITIVES:
        # bool is a subclass of int, so identity lookup must precede any
        # issubclass-based branch below.
        return {"type": _PRIMITIVES[hint]}

    origin = get_origin(hint)

    if origin is Literal:
        return _literal_schema(owner, field_name, hint)

    if origin is list or hint is list:
        args = get_args(hint)
        if not args:
            raise _unsupported(owner, field_name, hint)
        return {"type": "array", "items": _type_schema(owner, field_name, args[0])}

    if origin is dict or hint is dict:
        args = get_args(hint)
        if len(args) != 2:
            raise _unsupported(owner, field_name, hint)
        key_type, _ = _unwrap_annotated(args[0])
        if key_type is not str:
            raise ToolArgumentError(
                f"Cannot build a JSON Schema for {owner}.{field_name}: "
                f"mapping keys must be str, got {key_type!r}.",
                "Use dict[str, T] for this parameter; JSON object keys are always strings.",
            )
        return {
            "type": "object",
            "additionalProperties": _type_schema(owner, field_name, args[1]),
        }

    if _is_enum_type(hint):
        return _enum_schema(owner, field_name, hint)

    if _is_dataclass_type(hint):
        return build_object_schema(hint)

    raise _unsupported(owner, field_name, hint)


def _field_schema(owner: str, field_name: str, hint: object) -> tuple[dict[str, object], bool]:
    """Schema and optionality for one field, description included."""
    bare, description = _unwrap_annotated(hint)
    inner, optional = _split_optional(bare)
    inner_bare, inner_description = _unwrap_annotated(inner)
    schema = _type_schema(owner, field_name, inner_bare)
    text = description if description is not None else inner_description
    if text:
        schema["description"] = text
    return schema, optional


def build_object_schema(params_cls: type) -> dict[str, object]:
    """Derive the JSON Schema object for a dataclass of tool arguments.

    Field order is preserved, since providers surface properties to the model in
    the order given. A field is required exactly when it has no default, no
    default factory, and is not optional.

    Args:
        params_cls: A dataclass type whose fields are the tool's arguments.

    Returns:
        A schema with ``type``, ``properties``, ``required``, and
        ``additionalProperties: False``.

    Raises:
        ToolArgumentError: If ``params_cls`` is not a dataclass, or any field
            carries an annotation this module cannot express.
    """
    if not _is_dataclass_type(params_cls):
        raise ToolArgumentError(
            f"{params_cls!r} is not a dataclass, so no argument schema can be built.",
            "Declare the tool's Params as a dataclass.",
        )

    owner = params_cls.__name__
    try:
        hints = get_type_hints(params_cls, include_extras=True)
    except Exception as exc:  # unresolvable forward reference
        raise ToolArgumentError(
            f"Could not resolve type hints for {owner}: {exc}",
            "Make sure every annotation on this dataclass refers to a name "
            "importable at module level.",
        ) from exc

    properties: dict[str, object] = {}
    required: list[str] = []
    for spec in dataclasses.fields(params_cls):
        hint = hints.get(spec.name, spec.type)
        schema, optional = _field_schema(owner, spec.name, hint)
        properties[spec.name] = schema
        has_default = spec.default is not MISSING or spec.default_factory is not MISSING
        if not has_default and not optional:
            required.append(spec.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_TRUE_WORDS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "n", "off"})


def _describe(hint: object) -> str:
    """A short phrase naming what an annotation accepts, for error messages."""
    hint, _ = _unwrap_annotated(hint)
    if hint is str:
        return "a string"
    if hint is bool:
        return "a boolean"
    if hint is int:
        return "an integer"
    if hint is float:
        return "a number"

    origin = get_origin(hint)
    if origin is Literal:
        return "one of " + ", ".join(repr(choice) for choice in get_args(hint))
    if origin is list:
        args = get_args(hint)
        return f"an array of {_describe(args[0])}" if args else "an array"
    if origin is dict:
        args = get_args(hint)
        return f"an object whose values are {_describe(args[1])}" if args else "an object"
    if _is_enum_type(hint):
        return "one of " + ", ".join(repr(member.value) for member in hint)
    if _is_dataclass_type(hint):
        keys = ", ".join(spec.name for spec in dataclasses.fields(hint))
        return f"an object with keys {keys}"
    return f"a value of type {hint!r}"


def _reject(path: str, value: object, hint: object, instruction: str) -> ToolArgumentError:
    """Build the one error shape every coercion failure uses."""
    return ToolArgumentError(
        f"Argument {path!r}: expected {_describe(hint)}, received {_truncate(value)}.",
        instruction,
    )


def _coerce_bool(path: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise _reject(path, value, bool, f"Pass true or false for {path!r}, unquoted.")


def _coerce_int(path: str, value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise _reject(path, value, int, f"Pass a whole number for {path!r}, not a fraction.")
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError:
            pass
    raise _reject(path, value, int, f"Pass a whole number for {path!r}, unquoted.")


def _coerce_float(path: str, value: object) -> float:
    if isinstance(value, bool):
        raise _reject(path, value, float, f"Pass a number for {path!r}, not a boolean.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise _reject(path, value, float, f"Pass a number for {path!r}, unquoted.")


def _coerce_str(path: str, value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    raise _reject(
        path,
        value,
        str,
        f"Pass {path!r} as a JSON string; serialise structured data yourself if needed.",
    )


def _coerce_literal(path: str, hint: object, value: object) -> object:
    choices = get_args(hint)
    for choice in choices:
        if type(value) is type(choice) and value == choice:
            return choice
    # Models routinely quote enum members, so compare stringified forms before
    # giving up; the returned value is still the declared literal.
    for choice in choices:
        if isinstance(value, str | int | float | bool) and str(value).strip() == str(choice):
            return choice
    allowed = ", ".join(repr(choice) for choice in choices)
    raise _reject(path, value, hint, f"Set {path!r} to exactly one of: {allowed}.")


def _coerce_enum(path: str, hint: type[enum.Enum], value: object) -> enum.Enum:
    if isinstance(value, hint):
        return value
    for member in hint:
        if member.value == value or str(member.value) == str(value).strip():
            return member
        if member.name == str(value).strip():
            return member
    allowed = ", ".join(repr(member.value) for member in hint)
    raise _reject(path, value, hint, f"Set {path!r} to exactly one of: {allowed}.")


def _coerce(path: str, hint: object, value: object) -> object:
    """Coerce one JSON value to one annotation, or raise.

    ``path`` is the dotted route to this value from the top-level arguments
    object, so a failure inside a nested list or dataclass still names the exact
    key the model must fix.
    """
    hint, _ = _unwrap_annotated(hint)
    inner, optional = _split_optional(hint)
    if optional:
        if value is None:
            return None
        return _coerce(path, inner, value)

    if hint is bool:
        return _coerce_bool(path, value)
    if hint is int:
        return _coerce_int(path, value)
    if hint is float:
        return _coerce_float(path, value)
    if hint is str:
        return _coerce_str(path, value)

    origin = get_origin(hint)

    if origin is Literal:
        return _coerce_literal(path, hint, value)

    if origin is list:
        (item_hint,) = get_args(hint)
        if isinstance(value, str) or not isinstance(value, Sequence):
            # A single scalar where a list belongs is the most common model
            # slip; wrapping it is unambiguous, so accept it rather than
            # spending a turn on a correction.
            items: Sequence[object] = [value]
        else:
            items = value
        errors: list[ToolArgumentError] = []
        parsed: list[object] = []
        for index, item in enumerate(items):
            try:
                parsed.append(_coerce(f"{path}[{index}]", item_hint, item))
            except ToolArgumentError as exc:
                errors.append(exc)
        if errors:
            raise _merge_errors(errors)
        return parsed

    if origin is dict:
        _, value_hint = get_args(hint)
        if not isinstance(value, Mapping):
            raise _reject(path, value, hint, f"Pass {path!r} as a JSON object.")
        mapping_errors: list[ToolArgumentError] = []
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                mapping_errors.append(
                    _reject(path, key, hint, f"Use string keys in the {path!r} object.")
                )
                continue
            try:
                result[key] = _coerce(f"{path}.{key}", value_hint, item)
            except ToolArgumentError as exc:
                mapping_errors.append(exc)
        if mapping_errors:
            raise _merge_errors(mapping_errors)
        return result

    if _is_enum_type(hint):
        return _coerce_enum(path, hint, value)

    if _is_dataclass_type(hint):
        if not isinstance(value, Mapping):
            raise _reject(path, value, hint, f"Pass {path!r} as a JSON object.")
        return _parse_mapping(hint, value, prefix=f"{path}.")

    raise _reject(path, value, hint, f"Cannot interpret {path!r}; check the tool's schema.")


def _merge_errors(errors: Sequence[ToolArgumentError]) -> ToolArgumentError:
    """Fold several field errors into one, so the model sees them all at once.

    Reporting only the first would cost one round trip per bad field.
    """
    if len(errors) == 1:
        return errors[0]
    message = "\n".join(f"- {error.message}" for error in errors)
    hints: list[str] = []
    for error in errors:
        if error.hint and error.hint not in hints:
            hints.append(error.hint)
    return ToolArgumentError(
        f"{len(errors)} arguments were invalid:\n{message}",
        " ".join(hints) if hints else None,
    )


def _parse_mapping(params_cls: type, raw: Mapping[str, object], *, prefix: str = "") -> Any:
    """Validate ``raw`` against ``params_cls`` and construct it.

    Shared by :func:`parse_object` and nested-dataclass coercion; ``prefix``
    carries the dotted path of the enclosing field.
    """
    owner = params_cls.__name__
    try:
        hints = get_type_hints(params_cls, include_extras=True)
    except Exception as exc:
        raise ToolArgumentError(
            f"Could not resolve type hints for {owner}: {exc}",
            "This is a defect in the tool definition, not in your call.",
        ) from exc

    specs = dataclasses.fields(params_cls)
    known = {spec.name for spec in specs}
    errors: list[ToolArgumentError] = []

    unknown = [key for key in raw if key not in known]
    if unknown:
        listed = ", ".join(repr(key) for key in unknown)
        accepted = ", ".join(spec.name for spec in specs) or "(none)"
        errors.append(
            ToolArgumentError(
                f"Unknown argument(s) {listed} for {owner}.",
                f"Remove them and call again using only these arguments: {accepted}.",
            )
        )

    kwargs: dict[str, object] = {}
    for spec in specs:
        path = f"{prefix}{spec.name}"
        hint = hints.get(spec.name, spec.type)
        bare, _ = _unwrap_annotated(hint)
        _, optional = _split_optional(bare)
        has_default = spec.default is not MISSING or spec.default_factory is not MISSING

        if spec.name not in raw:
            if has_default:
                continue
            if optional:
                kwargs[spec.name] = None
                continue
            errors.append(
                ToolArgumentError(
                    f"Argument {path!r} is required but was not provided; "
                    f"it must be {_describe(bare)}.",
                    f"Call {owner} again with {path!r} included.",
                )
            )
            continue

        value = raw[spec.name]
        if value is None and not optional and has_default:
            # An explicit null for a defaulted field means "leave it alone".
            continue
        try:
            kwargs[spec.name] = _coerce(path, hint, value)
        except ToolArgumentError as exc:
            errors.append(exc)

    if errors:
        raise _merge_errors(errors)

    try:
        return params_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(
            f"Arguments for {owner} were rejected: {exc}",
            "Re-read the tool's schema and call it again with corrected arguments.",
        ) from exc


def parse_object(params_cls: type[T], raw: Mapping[str, object]) -> T:
    """Validate and coerce a model's JSON arguments into ``params_cls``.

    Lenient where models are reliably sloppy: ``"12"`` is accepted for an
    ``int``, ``"yes"``/``0``/``"TRUE"`` for a ``bool``, an ``int`` for a
    ``float``, and a bare scalar where a ``list[T]`` is expected. Strict
    otherwise: unknown keys, missing required keys, and out-of-set
    ``Literal``/``Enum`` values are all refused. Every field error in the payload
    is collected and reported together.

    Args:
        params_cls: The dataclass describing the tool's arguments.
        raw: The decoded JSON object the model supplied.

    Returns:
        An instance of ``params_cls``.

    Raises:
        ToolArgumentError: On any validation failure. The message names the
            field, the received value, and what was expected; the hint is phrased
            as an instruction, since it is fed back to the model verbatim.
    """
    if not _is_dataclass_type(params_cls):
        raise ToolArgumentError(
            f"{params_cls!r} is not a dataclass, so arguments cannot be parsed into it.",
            "This is a defect in the tool definition, not in your call.",
        )
    if not isinstance(raw, Mapping):
        raise ToolArgumentError(
            f"Expected a JSON object of arguments, received {_truncate(raw)}.",
            "Send the arguments as a JSON object mapping parameter names to values.",
        )
    return _parse_mapping(params_cls, raw)


_SECTION_RE = re.compile(
    r"^(?P<name>Args|Arguments|Parameters|Returns?|Yields?|Raises|Examples?|Notes?|Attributes)"
    r"\s*:\s*$",
    re.IGNORECASE,
)
_ARG_RE = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_]\w*)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<desc>.*)$"
)
_ARG_SECTIONS = frozenset({"args", "arguments", "parameters"})


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its summary and its ``Args:`` entries.

    Recognises ``name: description``, ``name (type): description``, and
    continuation lines indented past the entry they belong to. Everything before
    the first section header becomes the summary.

    Args:
        doc: A raw ``__doc__`` value, possibly ``None``.

    Returns:
        The summary text, and a mapping of parameter name to description.
    """
    if not doc or not doc.strip():
        return "", {}

    summary: list[str] = []
    entries: dict[str, list[str]] = {}
    seen_section = False
    in_args = False
    base_indent: int | None = None
    current: str | None = None

    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        header = _SECTION_RE.match(stripped)
        if header:
            seen_section = True
            in_args = header.group("name").lower() in _ARG_SECTIONS
            base_indent = None
            current = None
            continue
        if not in_args:
            # The summary is only what precedes the first section header; a
            # Returns:/Raises: body is neither summary nor argument text.
            if not seen_section:
                summary.append(line)
            continue
        if not stripped:
            current = None
            continue

        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent <= base_indent:
            match = _ARG_RE.match(stripped)
            if match:
                current = match.group("name").lstrip("*")
                entries.setdefault(current, [])
                text = match.group("desc").strip()
                if text:
                    entries[current].append(text)
                continue
        if current is not None:
            entries[current].append(stripped)

    return (
        "\n".join(summary).strip(),
        {name: " ".join(parts).strip() for name, parts in entries.items()},
    )


def is_context_annotation(hint: object) -> bool:
    """Whether an annotation refers to the host-injected ``ToolContext``.

    Handles the resolved class, an optional of it, and the bare string form left
    behind by ``from __future__ import annotations`` when resolution failed.
    """
    hint, _ = _unwrap_annotated(hint)
    inner, _ = _split_optional(hint)
    if isinstance(inner, str):
        return inner.split(".")[-1] == CONTEXT_TYPE_NAME
    return getattr(inner, "__name__", None) == CONTEXT_TYPE_NAME


def build_function_schema(fn: Callable[..., object]) -> tuple[dict[str, object], str]:
    """Derive an argument schema and a summary description from a function.

    ``self`` and any parameter annotated as ``ToolContext`` are omitted: they are
    supplied by the host, not by the model. Descriptions come from ``Doc``
    metadata when present, otherwise from the docstring's ``Args:`` block.

    Args:
        fn: A fully annotated callable.

    Returns:
        The JSON Schema object for the model-supplied parameters, and the
        docstring text preceding ``Args:``.

    Raises:
        ToolArgumentError: If a parameter is unannotated, uses an annotation this
            module cannot express, or is variadic.
    """
    owner = getattr(fn, "__name__", repr(fn))
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise ToolArgumentError(
            f"Could not resolve type hints for {owner}: {exc}",
            "Make sure every annotation on this function refers to a name "
            "importable at module level.",
        ) from exc

    summary, arg_docs = parse_docstring(inspect.getdoc(fn))
    signature = inspect.signature(fn)

    properties: dict[str, object] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ToolArgumentError(
                f"Cannot build a JSON Schema for {owner}: parameter {name!r} is variadic.",
                "Declare each tool parameter explicitly instead of using *args or **kwargs.",
            )
        if name not in hints:
            raise ToolArgumentError(
                f"Cannot build a JSON Schema for {owner}: parameter {name!r} is unannotated.",
                "Annotate every parameter of a tool function.",
            )

        hint = hints[name]
        if is_context_annotation(hint):
            continue

        schema, optional = _field_schema(owner, name, hint)
        if "description" not in schema and arg_docs.get(name):
            schema["description"] = arg_docs[name]
        properties[name] = schema
        if parameter.default is inspect.Parameter.empty and not optional:
            required.append(name)

    schema_object: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema_object, summary
