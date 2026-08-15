"""Unit tests for image validation."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.config.app_config import (
    AppConfig,
    CompetitionConfig,
    HardwareConfig,
    ImageConfig,
    LoggingConfig,
    PathsConfig,
    WorkerConfig,
)
from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from core.exceptions.vision import ValidationError
from core.utils.paths import project_root
from vision.validation.image_validator import ImageValidator


def _app_config() -> AppConfig:
    root = project_root()
    return AppConfig(
        app_name="Sentivis AI",
        app_version="1.0.0",
        logging=LoggingConfig("INFO", False, 1024, 1),
        image=ImageConfig(
            max_dimension=1024,
            max_file_size_bytes=5_000_000,
            yolo_inference_size=640,
            enhancement=DEFAULT_ENHANCEMENT_CONFIG,
        ),
        hardware=HardwareConfig(0.85, 0.90, True, 600),
        paths=PathsConfig(root / "cache", root / "exports", root / "logs", root / "models"),
        workers=WorkerConfig(2),
        competition=CompetitionConfig(0.55, 0.25, 42, 0.0, 64.0),
    )


def test_validate_accepts_valid_png(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    validator = ImageValidator(_app_config())
    result = validator.validate(path)
    assert result.width == 64
    assert result.height == 64


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    validator = ImageValidator(_app_config())
    with pytest.raises(ValidationError):
        validator.validate(tmp_path / "missing.png")
