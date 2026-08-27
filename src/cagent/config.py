"""Layered configuration.

Precedence, lowest to highest: built-in defaults, ``~/.cagent.toml``,
``.cagent.toml`` in the working directory, CLI overrides. Both config files use
a single flat ``[cagent]`` table whose keys are :class:`AgentConfig` field
names, so there is exactly one name to learn per setting no matter which layer
sets it.

There is deliberately no environment layer. It would restate every field under
a second spelling — ``CAGENT_MAX_STEPS`` beside ``max_steps`` — and a setting
with two spellings is a setting people have to check twice when the agent
behaves unexpectedly. The file is the one place a setting lives; a flag
overrides it for a single run.

``cwd`` is injected rather than read from globals so the whole chain is
unit-testable.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .errors import ConfigError

__all__ = [
    "AgentConfig",
    "ApprovalMode",
    "Wire",
    "load_config",
]

Wire = Literal["openai", "anthropic"]
"""Which request/response shape an endpoint speaks."""

ApprovalMode = Literal["suggest", "auto-edit", "full-auto"]
"""How much the agent may do without asking. See :class:`AgentConfig`."""

CONFIG_FILENAME = ".cagent.toml"
CONFIG_TABLE = "cagent"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass
class AgentConfig:
    """Every knob the agent reads, resolved into one object.

    An endpoint is described by four things, and the agent asks for exactly
    those: :attr:`base_url`, :attr:`model`, :attr:`api_key`, and :attr:`wire`.
    There is deliberately no notion of a named "provider". Shipping a table of
    vendors would mean shipping their model names, and model names go stale —
    a default of ``gpt-4o-mini`` or ``claude-3-5-sonnet`` is wrong within
    months and fails at the far end with an unhelpful 404. Only the person
    holding the key knows what their endpoint serves today, so the model name
    is required rather than guessed.

    Approval modes:

    * ``suggest`` - confirm every mutation, file writes included.
    * ``auto-edit`` - file writes inside the workspace run free; shell commands
      and anything else mutating still ask.
    * ``full-auto`` - only :attr:`~cagent.types.RiskLevel.DANGEROUS` calls ask.

    ``api_key`` is masked by :meth:`__repr__`, so logging a config, dumping it
    into a trace, or letting it surface in a traceback will not leak it.
    """

    # endpoint
    base_url: str | None = None
    """Where to send requests, e.g. ``https://api.example.com/v1``. Used
    verbatim, with the wire's path appended."""

    model: str | None = None
    """The model name the endpoint expects. No default: see the class docstring."""

    api_key: str | None = None
    """The endpoint's key. It comes from a config file only — ``.cagent.toml``
    is untracked by default, and there is no flag, because a key on the command
    line lands in shell history and in the process list."""

    wire: Wire = "openai"
    """Request format. ``openai`` (Chat Completions) suits almost everything;
    ``anthropic`` is for the Messages API."""

    requires_key: bool = True
    """Set false for a local server such as Ollama or llama.cpp, which would
    otherwise be rejected for having no key."""

    temperature: float = 0.0
    max_output_tokens: int = 8192
    request_timeout: float = 120.0
    max_retries: int = 4

    prices: dict[str, object] = field(default_factory=dict)
    """Optional per-model rates, for reporting session cost. No rates ship with
    the project: a built-in table goes stale, and a stale price reports a
    confidently wrong figure. See :mod:`cagent.cli.pricing`."""

    # context
    context_window: int = 128_000
    compact_threshold: float = 0.75
    keep_recent_turns: int = 6
    tool_output_head_lines: int = 60
    tool_output_tail_lines: int = 40
    tool_output_max_chars: int = 20_000

    # loop
    max_steps: int = 40
    token_budget: int | None = None
    max_repeated_calls: int = 3
    bash_timeout: float = 120.0

    # edit
    fuzzy_threshold: float = 0.86

    # safety
    approval_mode: ApprovalMode = "auto-edit"
    workspace: Path = field(default_factory=Path.cwd)
    allow_outside_workspace: bool = False

    # repomap
    repo_map_token_budget: int = 1600
    repo_map_enabled: bool = True

    # observability
    trace_dir: Path | None = None
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Path fields are absolute from here on, so every sandbox check can
        # compare them without re-resolving.
        self.workspace = Path(self.workspace).expanduser().resolve()
        if self.trace_dir is not None:
            self.trace_dir = Path(self.trace_dir).expanduser().resolve()

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{f.name}={'***' if f.name == 'api_key' and self.api_key else getattr(self, f.name)!r}"
            for f in fields(self)
        )
        return f"{type(self).__name__}({shown})"

    @property
    def resolved_base_url(self) -> str:
        """The endpoint, without a trailing slash.

        Raises:
            ConfigError: If no ``base_url`` was configured. There is nothing
                sensible to fall back to — guessing a vendor would be guessing
                whose key the user holds.
        """
        if not self.base_url:
            raise ConfigError(
                "No base_url configured. Set base_url in .cagent.toml, or pass "
                "--base-url, to the endpoint you want to call "
                "(for example https://api.example.com/v1)."
            )
        return self.base_url.rstrip("/")

    @property
    def resolved_model(self) -> str:
        """The model name to request.

        Raises:
            ConfigError: If no ``model`` was configured. A built-in default
                would be a model name frozen at release time, and the failure
                it produces once the vendor retires it is a remote 404 that
                says nothing useful.
        """
        if not self.model:
            raise ConfigError(
                "No model configured. Set model in .cagent.toml, or pass "
                "--model, to a model your endpoint serves."
            )
        return self.model

    @property
    def model_for_tokens(self) -> str:
        """The model name for token estimation only, empty when unset.

        Estimation uses the name solely to choose an encoder and already
        degrades to a heuristic, so measuring context pressure must not require
        a configured model — that would make a budgeting concern fail for a
        reason that belongs to sending a request.
        """
        return self.model or ""

    @property
    def resolved_wire(self) -> Wire:
        """The request format. Defaults to the OpenAI shape."""
        return self.wire

    @property
    def compact_at_tokens(self) -> int:
        """Prompt size at which the context manager starts compacting."""
        return int(self.context_window * self.compact_threshold)

    def validate(self) -> AgentConfig:
        """Check internal consistency, returning ``self`` so calls can chain.

        Raises:
            ConfigError: on a missing endpoint, model, or required key, an
                out-of-range threshold, or a non-positive limit.
        """
        _ = self.resolved_base_url  # raises with its own message if unset
        _ = self.resolved_model

        if not self.api_key and self.requires_key:
            raise ConfigError(
                f"No API key for the endpoint {self.resolved_base_url!r}. Set "
                "api_key in .cagent.toml (untracked by default) or in "
                "~/.cagent.toml. For a local server that needs no key, pass "
                "--no-key or set requires_key = false."
            )

        for name in ("compact_threshold", "fuzzy_threshold"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must be strictly between 0 and 1, got {value!r}.")

        positive_ints = (
            "max_output_tokens",
            "context_window",
            "keep_recent_turns",
            "tool_output_head_lines",
            "tool_output_tail_lines",
            "tool_output_max_chars",
            "max_steps",
            "max_repeated_calls",
            "repo_map_token_budget",
        )
        for name in positive_ints:
            value_int = int(getattr(self, name))
            if value_int <= 0:
                raise ConfigError(f"{name} must be positive, got {value_int!r}.")

        for name in ("request_timeout", "bash_timeout"):
            value_float = float(getattr(self, name))
            if value_float <= 0:
                raise ConfigError(f"{name} must be positive, got {value_float!r}.")

        if self.max_retries < 0:
            raise ConfigError(f"max_retries must not be negative, got {self.max_retries!r}.")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ConfigError(f"token_budget must be positive when set, got {self.token_budget!r}.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be within [0, 2], got {self.temperature!r}.")
        if self.approval_mode not in ("suggest", "auto-edit", "full-auto"):
            raise ConfigError(
                f"approval_mode must be one of suggest, auto-edit, full-auto; "
                f"got {self.approval_mode!r}."
            )
        return self


_HINTS: dict[str, object] | None = None


def _hints() -> Mapping[str, object]:
    """Resolved annotations for :class:`AgentConfig`, computed once."""
    global _HINTS
    if _HINTS is None:
        _HINTS = dict(get_type_hints(AgentConfig))
    return _HINTS


def _coerce(label: str, raw: object, hint: object) -> object:
    """Convert one layer's raw value to the field's annotated type.

    ``label`` names the source (a file key with its path, or a flag) and appears verbatim
    in any :class:`ConfigError`, so a bad value is traceable to where it was set.
    """
    origin = get_origin(hint)

    if origin is UnionType or origin is Union:
        args = get_args(hint)
        concrete = [arg for arg in args if arg is not type(None)]
        optional = len(concrete) < len(args)
        if optional and isinstance(raw, str) and not raw.strip():
            return None
        for arg in concrete:
            try:
                return _coerce(label, raw, arg)
            except ConfigError:
                continue
        raise ConfigError(f"Cannot interpret {label}={raw!r}.")

    if origin is Literal:
        choices = get_args(hint)
        text = str(raw).strip()
        if text in choices:
            return text
        allowed = ", ".join(str(choice) for choice in choices)
        raise ConfigError(f"{label} must be one of {allowed}; got {raw!r}.")

    text = str(raw).strip()
    if hint is bool:
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigError(f"{label} must be a boolean (1/true/yes/on); got {raw!r}.")
    if hint is int:
        try:
            return int(text)
        except ValueError:
            raise ConfigError(f"{label} must be an integer; got {raw!r}.") from None
    if hint is float:
        try:
            return float(text)
        except ValueError:
            raise ConfigError(f"{label} must be a number; got {raw!r}.") from None
    if hint is str:
        return text
    if hint is Path:
        return Path(text).expanduser()
    raise ConfigError(f"Unsupported configuration type for {label}: {hint!r}.")


def _read_config_file(path: Path) -> Mapping[str, object]:
    """Parse the ``[cagent]`` table from one TOML file; ``{}`` if absent."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc

    table = document.get(CONFIG_TABLE)
    if table is None:
        if document:
            raise ConfigError(
                f"Config file {path} must place settings in a [{CONFIG_TABLE}] table."
            )
        return {}
    if not isinstance(table, dict):
        raise ConfigError(f"[{CONFIG_TABLE}] in {path} must be a table.")
    return table


def load_config(
    cli_overrides: Mapping[str, object] | None = None,
    *,
    cwd: Path | None = None,
) -> AgentConfig:
    """Merge every configuration layer into one :class:`AgentConfig`.

    Later layers win: home file, then project file, then ``cli_overrides``
    (``None`` values there mean "flag not passed" and are ignored).

    The result is not validated; call :meth:`AgentConfig.validate` when the
    endpoint is actually about to be called.

    Raises:
        ConfigError: on an unknown setting name or an uninterpretable value.
    """
    base = Path.cwd() if cwd is None else Path(cwd)
    hints = _hints()
    known = {spec.name for spec in fields(AgentConfig)}
    values: dict[str, object] = {}

    for path in (Path.home() / CONFIG_FILENAME, base / CONFIG_FILENAME):
        for key, raw in _read_config_file(path).items():
            if key not in known:
                raise ConfigError(f"Unknown setting {key!r} in {path}.")
            values[key] = _coerce(f"{key} (in {path})", raw, hints[key])

    for key, raw in (cli_overrides or {}).items():
        if key not in known:
            raise ConfigError(f"Unknown setting {key!r} in CLI overrides.")
        if raw is None:
            continue
        values[key] = _coerce(f"--{key.replace('_', '-')}", raw, hints[key])

    values.setdefault("workspace", base)
    return AgentConfig(**values)  # type: ignore[arg-type]


