"""Structured multi-channel logging."""

import logging
import sys
from pathlib import Path

from core.config.app_config import LoggingConfig
from core.logging.formatters import StructuredTextFormatter
from core.logging.handlers import ChannelFilter, build_rotating_handler

_CONFIGURED = False


def reset_logging_for_tests() -> None:
    """Reset logging state so tests can reconfigure handlers."""
    global _CONFIGURED
    _CONFIGURED = False
    logging.getLogger("sentivis").handlers.clear()


def configure_logging(config: LoggingConfig, log_dir: Path) -> None:
    """Configure application, pipeline, error, and benchmark log channels."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, config.level.upper(), logging.INFO)

    root = logging.getLogger("sentivis")
    root.setLevel(level)
    root.handlers.clear()

    text_formatter = StructuredTextFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_handler = build_rotating_handler(log_dir / "application.log", config)
    app_handler.setFormatter(text_formatter)
    root.addHandler(app_handler)

    pipeline_handler = build_rotating_handler(log_dir / "pipeline.log", config)
    pipeline_handler.addFilter(ChannelFilter("sentivis.services.pipeline"))
    pipeline_handler.setFormatter(text_formatter)
    root.addHandler(pipeline_handler)

    error_handler = build_rotating_handler(log_dir / "error.log", config, level=logging.ERROR)
    error_handler.setFormatter(text_formatter)
    root.addHandler(error_handler)

    benchmark_handler = build_rotating_handler(log_dir / "benchmark.log", config)
    benchmark_handler.addFilter(ChannelFilter("sentivis.benchmark"))
    benchmark_handler.setFormatter(text_formatter)
    root.addHandler(benchmark_handler)

    if config.console_enabled:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(text_formatter)
        root.addHandler(console)

    _CONFIGURED = True
    root.info("Logging configured at level %s with rotating channels", config.level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the sentivis hierarchy."""
    if name.startswith("sentivis."):
        return logging.getLogger(name)
    return logging.getLogger(f"sentivis.{name}")
