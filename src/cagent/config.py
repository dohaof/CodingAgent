"""Layered configuration.

Precedence, lowest to highest: built-in defaults, ``~/.cagent.toml``,
``.cagent.toml`` in the working directory, ``CAGENT_*`` environment variables,
CLI overrides. Both config files use a single flat ``[cagent]`` table whose keys
are :class:`AgentConfig` field names, so there is exactly one name to learn per
setting no matter which layer sets it.

``env`` and ``cwd`` are injected rather than read from globals so the whole
chain is unit-testable.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .errors import ConfigError

__all__ = [
    "PRESETS",
    "AgentConfig",
    "ApprovalMode",
    "ProviderPreset",
    "Wire",
    "load_config",
]

Wire = Literal["openai", "anthropic"]
"""Which request/response shape a provider speaks."""

ApprovalMode = Literal["suggest", "auto-edit", "full-auto"]
"""How much the agent may do without asking. See :class:`AgentConfig`."""

CONFIG_FILENAME = ".cagent.toml"
CONFIG_TABLE = "cagent"
ENV_PREFIX = "CAGENT_"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Defaults for a known vendor endpoint.

    ``env_key`` is that vendor's conventional key variable, checked after
    ``CAGENT_API_KEY``. ``requires_key`` is false for local servers.
    """

    name: str
    base_url: str
    default_model: str
    wire: Wire
    env_key: str
    requires_key: bool = True


PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        wire="openai",
        env_key="DEEPSEEK_API_KEY",
    ),
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        wire="openai",
        env_key="OPENAI_API_KEY",
    ),
    "anthropic": ProviderPreset(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-20250514",
        wire="anthropic",
        env_key="ANTHROPIC_API_KEY",
    ),
    "moonshot": ProviderPreset(
        name="moonshot",
        base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k2-0905-preview",
        wire="openai",
        env_key="MOONSHOT_API_KEY",
    ),
    "dashscope": ProviderPreset(
        name="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        wire="openai",
        env_key="DASHSCOPE_API_KEY",
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        wire="openai",
        env_key="OPENROUTER_API_KEY",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5-coder",
        wire="openai",
        env_key="OLLAMA_API_KEY",
        requires_key=False,
    ),
}

@dataclass
class AgentConfig:
    """Every knob the agent reads, resolved into one object.

    Approval modes:

    * ``suggest`` - confirm every mutation, file writes included.
    * ``auto-edit`` - file writes inside the workspace run free; shell commands
      and anything else mutating still ask.
    * ``full-auto`` - only :attr:`~cagent.types.RiskLevel.DANGEROUS` calls ask.

    ``api_key`` is masked by :meth:`__repr__`, so logging a config, dumping it
    into a trace, or letting it surface in a traceback will not leak it.
    """

    # provider
    provider: str = "deepseek"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    wire: Wire | None = None
    temperature: float = 0.0
    max_output_tokens: int = 8192
    request_timeout: float = 120.0
    max_retries: int = 4

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
    def preset(self) -> ProviderPreset | None:
        """The matching preset, or ``None`` for an unknown provider name."""
        return PRESETS.get(self.provider)

    @property
    def resolved_base_url(self) -> str:
        """Explicit ``base_url`` if set, else the preset's."""
        if self.base_url:
            return self.base_url.rstrip("/")
        preset = self.preset
        if preset is None:
            raise ConfigError(
                f"Unknown provider {self.provider!r} and no base_url given. "
                f"Known providers: {', '.join(sorted(PRESETS))}."
            )
        return preset.base_url

    @property
    def resolved_model(self) -> str:
        """Explicit ``model`` if set, else the preset's default."""
        if self.model:
            return self.model
        preset = self.preset
        if preset is None:
            raise ConfigError(
                f"Unknown provider {self.provider!r} and no model given. "
                f"Known providers: {', '.join(sorted(PRESETS))}."
            )
        return preset.default_model

    @property
    def resolved_wire(self) -> Wire:
        """Explicit ``wire`` if set, else the preset's; unknown providers
        default to the OpenAI shape, which most vendors emulate."""
        if self.wire:
            return self.wire
        preset = self.preset
        return preset.wire if preset is not None else "openai"

    @property
    def compact_at_tokens(self) -> int:
        """Prompt size at which the context manager starts compacting."""
        return int(self.context_window * self.compact_threshold)

    def validate(self) -> AgentConfig:
        """Check internal consistency, returning ``self`` so calls can chain.

        Raises:
            ConfigError: on a missing required key, an out-of-range threshold,
                or a non-positive limit.
        """
        preset = self.preset
        if not self.api_key and (preset is None or preset.requires_key):
            hint = f" Set CAGENT_API_KEY or {preset.env_key}." if preset else ""
            raise ConfigError(f"No API key for provider {self.provider!r}.{hint}")

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

    ``label`` names the source (env var, file key, or flag) and appears verbatim
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


def _resolve_vendor_key(provider: str, environ: Mapping[str, str]) -> str | None:
    """Fall back to the preset's conventional key variable."""
    preset = PRESETS.get(provider)
    if preset is None:
        return None
    return environ.get(preset.env_key) or None


def load_config(
    cli_overrides: Mapping[str, object] | None = None,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AgentConfig:
    """Merge every configuration layer into one :class:`AgentConfig`.

    Later layers win: home file, then project file, then ``CAGENT_*`` env vars,
    then ``cli_overrides`` (``None`` values there mean "flag not passed" and are
    ignored). The API key is resolved last, from ``CAGENT_API_KEY`` and then the
    provider preset's own variable, because that lookup depends on the provider
    every earlier layer just agreed on.

    The result is not validated; call :meth:`AgentConfig.validate` when a live
    provider is actually needed.

    Raises:
        ConfigError: on an unknown setting name or an uninterpretable value.
    """
    environ = os.environ if env is None else env
    base = Path.cwd() if cwd is None else Path(cwd)
    hints = _hints()
    known = {spec.name for spec in fields(AgentConfig)}
    values: dict[str, object] = {}

    for path in (Path.home() / CONFIG_FILENAME, base / CONFIG_FILENAME):
        for key, raw in _read_config_file(path).items():
            if key not in known:
                raise ConfigError(f"Unknown setting {key!r} in {path}.")
            values[key] = _coerce(f"{key} (in {path})", raw, hints[key])

    for key in sorted(known):
        name = f"{ENV_PREFIX}{key.upper()}"
        raw_env = environ.get(name)
        if raw_env is not None:
            values[key] = _coerce(name, raw_env, hints[key])

    for key, raw in (cli_overrides or {}).items():
        if key not in known:
            raise ConfigError(f"Unknown setting {key!r} in CLI overrides.")
        if raw is None:
            continue
        values[key] = _coerce(f"--{key.replace('_', '-')}", raw, hints[key])

    values.setdefault("workspace", base)
    config = AgentConfig(**values)  # type: ignore[arg-type]
    if not config.api_key:
        config.api_key = _resolve_vendor_key(config.provider, environ)
    return config


