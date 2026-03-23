from __future__ import annotations

import os

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_CLASS = "Qwen/Qwen3-0.6B"
DEFAULT_DB_PATH = "~/.inferlib/chats.db"
DEFAULT_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean for {field_name}: {value!r}")


def _parse_int(value: object, *, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Invalid integer for {field_name}: {value!r}")


def _parse_str(value: object, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"Invalid string for {field_name}: {value!r}")


@dataclass(slots=True, frozen=True)
class AppSettings:
    model_class: str = DEFAULT_MODEL_CLASS
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path(DEFAULT_DB_PATH).expanduser().resolve()
    page_size: int = 32
    batch_size: int = 4
    max_active_sequences: int = 8
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES
    log_level: str = "INFO"
    log_format: str = "text"
    serve_ui: bool = True

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("port must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_active_sequences <= 0:
            raise ValueError("max_active_sequences must be positive")
        if self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if self.log_format not in {"text", "json"}:
            raise ValueError("log_format must be 'text' or 'json'")
        object.__setattr__(self, "db_path", self.db_path.expanduser().resolve())
        object.__setattr__(self, "log_level", self.log_level.upper())
        object.__setattr__(self, "log_format", self.log_format.lower())

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["db_path"] = str(self.db_path)
        return data


def load_settings(
    *,
    overrides: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    env = dict(os.environ if environ is None else environ)
    values: dict[str, object] = {}
    overrides = dict(overrides or {})

    def env_value(name: str, *, legacy_name: str | None = None) -> str | None:
        if name in env and env[name] != "":
            return env[name]
        if legacy_name and legacy_name in env and env[legacy_name] != "":
            return env[legacy_name]
        return None

    def get_value(
        field_name: str,
        *,
        env_name: str,
        parser,
        default: object,
        legacy_env_name: str | None = None,
    ) -> object:
        if field_name in overrides and overrides[field_name] is not None:
            return parser(overrides[field_name], field_name=field_name)
        raw = env_value(env_name, legacy_name=legacy_env_name)
        if raw is not None:
            return parser(raw, field_name=field_name)
        return default

    values["model_class"] = get_value(
        "model_class",
        env_name="INFERLIB_MODEL_CLASS",
        parser=_parse_str,
        default=DEFAULT_MODEL_CLASS,
    )
    values["host"] = get_value(
        "host",
        env_name="INFERLIB_HOST",
        parser=_parse_str,
        default="0.0.0.0",
    )
    values["port"] = get_value(
        "port",
        env_name="INFERLIB_PORT",
        parser=_parse_int,
        default=8000,
    )
    values["db_path"] = Path(
        get_value(
            "db_path",
            env_name="INFERLIB_DB_PATH",
            legacy_env_name="DB_PATH",
            parser=_parse_str,
            default=DEFAULT_DB_PATH,
        )
    )
    values["page_size"] = get_value(
        "page_size",
        env_name="INFERLIB_PAGE_SIZE",
        parser=_parse_int,
        default=32,
    )
    values["batch_size"] = get_value(
        "batch_size",
        env_name="INFERLIB_BATCH_SIZE",
        parser=_parse_int,
        default=4,
    )
    values["max_active_sequences"] = get_value(
        "max_active_sequences",
        env_name="INFERLIB_MAX_ACTIVE_SEQUENCES",
        parser=_parse_int,
        default=8,
    )
    values["memory_limit_bytes"] = get_value(
        "memory_limit_bytes",
        env_name="INFERLIB_MEMORY_LIMIT_BYTES",
        parser=_parse_int,
        default=DEFAULT_MEMORY_LIMIT_BYTES,
    )
    values["log_level"] = get_value(
        "log_level",
        env_name="INFERLIB_LOG_LEVEL",
        parser=_parse_str,
        default="INFO",
    )
    values["log_format"] = get_value(
        "log_format",
        env_name="INFERLIB_LOG_FORMAT",
        parser=_parse_str,
        default="text",
    )
    values["serve_ui"] = get_value(
        "serve_ui",
        env_name="INFERLIB_UI",
        parser=_parse_bool,
        default=True,
    )
    return AppSettings(**values)
