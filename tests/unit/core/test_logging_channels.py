"""Unit tests for structured logging channels."""

from pathlib import Path

from core.config.loader import load_app_config
from core.logging.logger_factory import configure_logging, get_logger, reset_logging_for_tests


def test_configure_logging_creates_rotating_channels(tmp_path: Path) -> None:
    reset_logging_for_tests()
    app_config = load_app_config()
    configure_logging(app_config.logging, tmp_path)
    logger = get_logger("services.pipeline.orchestrator")
    logger.info("pipeline channel test")
    error_logger = get_logger("app.bootstrap")
    error_logger.error("error channel test")
    assert (tmp_path / "application.log").exists()
    assert (tmp_path / "pipeline.log").exists()
    assert (tmp_path / "error.log").exists()
    assert (tmp_path / "benchmark.log").exists()
