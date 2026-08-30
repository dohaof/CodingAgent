"""Tool registration, lookup, and the function-style declaration path.

:class:`ToolRegistry` owns the mapping the agent loop dispatches through and the
:class:`~cagent.types.ToolSpec` list the provider is shown. The two are
deliberately separable: a disabled tool disappears from :meth:`ToolRegistry.specs`
so the model stops calling it, yet stays retrievable by name so an in-flight call
from an earlier turn still resolves instead of becoming a confusing error.

The :func:`tool` decorator is the lightweight alternative to subclassing
:class:`~cagent.tools.base.BaseTool`: it synthesises the arguments dataclass from
an annotated signature, so a small tool is one function with a docstring.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, get_type_hints

from ..errors import ToolNotFoundError
from ..types import RiskLevel, ToolSpec
from .base import BaseTool, ToolContext, ToolOutcome
from .schema import Doc, is_context_annotation, parse_docstring

__all__ = ["ToolRegistry", "default_registry", "tool"]

_SUGGESTION_CUTOFF = 0.6
_MAX_SUGGESTIONS = 3


class ToolRegistry:
    """A named collection of tool instances.

    Insertion order is preserved, since it determines the order tools are
    presented to the model.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._disabled: set[str] = set()

    def register(self, tool_instance: BaseTool) -> None:
        """Add a tool instance.

        Raises:
            ValueError: If the tool is unnamed, or the name is already taken.
                Silently replacing a tool would make dispatch depend on import
                order.
        """
        name = tool_instance.name
        if not name:
            raise ValueError(f"{type(tool_instance).__name__} must define a non-empty name.")
        if name in self._tools:
            existing = type(self._tools[name]).__name__
            raise ValueError(
                f"A tool named {name!r} is already registered "
                f"({existing} vs {type(tool_instance).__name__})."
            )
        self._tools[name] = tool_instance

    def register_class(self, cls: type[BaseTool]) -> BaseTool:
        """Instantiate ``cls`` with no arguments and register it."""
        instance = cls()
        self.register(instance)
        return instance

    def get(self, name: str) -> BaseTool:
        """Look up a tool by exact name, enabled or not.

        Raises:
            ToolNotFoundError: If the name is unknown. The nearest registered
                names are attached as a hint, which is usually enough for the
                model to correct a typo or a hallucinated tool on its next turn.
        """
        found = self._tools.get(name)
        if found is not None:
            return found
        close = difflib.get_close_matches(
            name, list(self._tools), n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
        )
        raise ToolNotFoundError(name, close or sorted(self._tools))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Every registered name, in registration order."""
        return list(self._tools)

    def enabled_names(self) -> list[str]:
        """Names currently advertised to the model."""
        return [name for name in self._tools if name not in self._disabled]

    def specs(self) -> list[ToolSpec]:
        """Specs for the enabled tools, in registration order."""
        return [
            instance.spec()
            for name, instance in self._tools.items()
            if name not in self._disabled
        ]

    def enable(self, name: str) -> None:
        """Re-advertise a previously disabled tool."""
        self.get(name)
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        """Stop advertising a tool without removing it."""
        self.get(name)
        self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        """Whether ``name`` is registered and currently advertised."""
        return name in self._tools and name not in self._disabled

    def subset(self, names: Sequence[str]) -> ToolRegistry:
        """A new registry holding only ``names``, sharing the same instances.

        Raises:
            ToolNotFoundError: If any name is not registered here.
        """
        child = ToolRegistry()
        for name in names:
            child.register(self.get(name))
        return child


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    risk: RiskLevel = RiskLevel.SAFE,
    parallel_safe: bool = False,
) -> Callable[[Callable[..., ToolOutcome | str]], type[BaseTool]]:
    """Turn an annotated function into a :class:`BaseTool` subclass.

    The arguments dataclass is synthesised from the signature, defaults and
    ``Annotated`` documentation included; a parameter annotated as
    :class:`~cagent.tools.base.ToolContext` is injected at call time rather than
    requested from the model. The name and description default to the function's
    own name and docstring summary.

    Returns the tool *class*, so the result can be registered like any other.

    Args:
        name: Overrides the function name.
        description: Overrides the docstring summary.
        risk: Approval class for the synthesised tool.
        parallel_safe: Whether independent calls may run concurrently.
    """

    def decorate(fn: Callable[..., ToolOutcome | str]) -> type[BaseTool]:
        fn_name = getattr(fn, "__name__", "tool")
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception as exc:
            raise ValueError(f"Could not resolve type hints for {fn_name}: {exc}") from exc

        summary, arg_docs = parse_docstring(fn.__doc__)
        params_cls, context_params = _build_params(fn, fn_name, hints, arg_docs)
        field_names = [spec.name for spec in dataclasses.fields(params_cls)]

        namespace: dict[str, Any] = {
            "name": name or fn_name,
            "description": (description or summary or f"Run {fn_name}.").strip(),
            "risk": risk,
            "parallel_safe": parallel_safe,
            "Params": params_cls,
            "__doc__": fn.__doc__ or f"Tool wrapper around {fn_name}.",
            "__module__": getattr(fn, "__module__", __name__),
            "_fn": staticmethod(fn),
            "_field_names": tuple(field_names),
            "_context_params": tuple(context_params),
            "run": _make_run(),
        }
        return type(f"{_class_name(name or fn_name)}Tool", (_FunctionTool,), namespace)

    return decorate


def _class_name(raw: str) -> str:
    """``read_file`` -> ``ReadFile``, for a readable synthesised class name."""
    return "".join(part.capitalize() for part in raw.replace("-", "_").split("_") if part)


def _build_params(
    fn: Callable[..., object],
    fn_name: str,
    hints: dict[str, Any],
    arg_docs: dict[str, str],
) -> tuple[type, list[str]]:
    """Synthesise the arguments dataclass, and list the injected parameters."""
    import inspect

    signature = inspect.signature(fn)
    fields: list[tuple[str, Any, Any]] = []
    context_params: list[str] = []

    for param_name, parameter in signature.parameters.items():
        if param_name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError(
                f"{fn_name}: tool functions cannot take *args or **kwargs; "
                f"declare each parameter explicitly."
            )
        if param_name not in hints:
            raise ValueError(f"{fn_name}: parameter {param_name!r} must be annotated.")

        annotation = hints[param_name]
        if is_context_annotation(annotation):
            context_params.append(param_name)
            continue

        if param_name in arg_docs and not _has_doc(annotation):
            # Fold the docstring text into the annotation so the synthesised
            # dataclass carries it, and build_object_schema needs no second source.
            annotation = _annotate(annotation, arg_docs[param_name])

        if parameter.default is inspect.Parameter.empty:
            fields.append((param_name, annotation, dataclasses.MISSING))
        elif isinstance(parameter.default, list | dict | set):
            # A mutable default becomes a copying factory, so one call cannot
            # mutate the default seen by the next.
            fields.append(
                (
                    param_name,
                    annotation,
                    dataclasses.field(default_factory=_copy_factory(parameter.default)),
                )
            )
        else:
            fields.append((param_name, annotation, dataclasses.field(default=parameter.default)))

    ordered: list[Any] = [
        (field_name, annotation)
        if default is dataclasses.MISSING
        else (field_name, annotation, default)
        for field_name, annotation, default in fields
    ]
    params_cls = dataclasses.make_dataclass(
        f"{_class_name(fn_name)}Params",
        ordered,
        frozen=True,
        slots=True,
    )
    params_cls.__module__ = getattr(fn, "__module__", __name__)
    return params_cls, context_params


def _copy_factory(value: list[Any] | dict[Any, Any] | set[Any]) -> Callable[[], Any]:
    """A default_factory returning a fresh copy of a mutable default."""

    def factory() -> Any:
        return type(value)(value)

    return factory


def _has_doc(annotation: object) -> bool:
    """Whether an annotation already carries ``Doc`` or string metadata."""
    metadata = getattr(annotation, "__metadata__", ())
    return any(isinstance(meta, Doc | str) for meta in metadata)


def _annotate(annotation: Any, text: str) -> Any:
    """Attach a description to an annotation as ``Annotated`` metadata."""
    from typing import Annotated

    return Annotated[annotation, Doc(text)]


class _FunctionTool(BaseTool):
    """Base for the classes :func:`tool` synthesises."""

    _fn: ClassVar[Callable[..., ToolOutcome | str]]
    _field_names: ClassVar[tuple[str, ...]]
    _context_params: ClassVar[tuple[str, ...]]

    def run(self, params: Any, ctx: ToolContext) -> ToolOutcome:
        raise NotImplementedError


def _make_run() -> Callable[[Any, Any, ToolContext], ToolOutcome]:
    """Build the ``run`` that unpacks params back into keyword arguments."""

    def run(self: Any, params: Any, ctx: ToolContext) -> ToolOutcome:
        kwargs: dict[str, object] = {
            field_name: getattr(params, field_name) for field_name in self._field_names
        }
        for param_name in self._context_params:
            kwargs[param_name] = ctx
        result = type(self)._fn(**kwargs)
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome.ok("" if result is None else str(result))

    return run


def default_registry() -> ToolRegistry:
    """Build the registry the CLI runs with.

    Concrete tool modules are imported here rather than at module scope so that
    importing the registry stays cheap, and each group is optional: a module the
    later layer has not written yet is skipped instead of breaking startup.

    Intended contents: ``read_file``, ``write_file``, ``edit_file``,
    ``multi_edit``, ``list_dir``, ``glob_files``, ``grep_search``, ``run_bash``,
    ``apply_patch`` (optional), and ``finish``.
    """
    registry = ToolRegistry()

    groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("files", ("ReadFileTool", "WriteFileTool", "ListDirTool")),
        ("edit", ("EditFileTool", "MultiEditTool")),
        ("search", ("GlobFilesTool", "GrepSearchTool")),
        ("shell", ("RunBashTool",)),
        ("patch", ("ApplyPatchTool",)),
        ("control", ("FinishTool",)),
    )

    for module_name, class_names in groups:
        try:
            module = __import__(f"{__package__}.{module_name}", fromlist=list(class_names))
        except ImportError:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if isinstance(cls, type) and issubclass(cls, BaseTool) and cls.name not in registry:
                registry.register_class(cls)

    return registry
