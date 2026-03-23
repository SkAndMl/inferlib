from __future__ import annotations

import json
import logging
import sys
import traceback

from datetime import datetime, UTC
from typing import Any


_CONFIGURED = False

_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def _record_extra(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
            continue
        extras[key] = value
    return extras


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = _record_extra(record)
        if extras:
            extra_str = " ".join(f"{key}={extras[key]!r}" for key in sorted(extras))
            message = f"{message} {extra_str}"
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_record_extra(record))
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", fmt: str = "text") -> None:
    global _CONFIGURED

    formatter: logging.Formatter
    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.handlers = [handler]
    root.setLevel(level.upper())

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(level.upper())

    _CONFIGURED = True


def ensure_logging_configured(*, level: str = "INFO", fmt: str = "text") -> None:
    if not _CONFIGURED:
        configure_logging(level=level, fmt=fmt)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
