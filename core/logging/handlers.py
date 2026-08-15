"""Channel-specific logging handlers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config.app_config import LoggingConfig
from core.logging.formatters import StructuredTextFormatter


def build_rotating_handler(
    path: Path,
    config: LoggingConfig,
    *,
    level: int = logging.NOTSET,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=config.max_file_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        StructuredTextFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    if level != logging.NOTSET:
        handler.setLevel(level)
    return handler


class ChannelFilter(logging.Filter):
    """Route records to a dedicated logger namespace."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefix)
