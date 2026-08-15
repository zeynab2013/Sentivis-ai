"""Unit tests for image validator edge cases."""

from pathlib import Path

import pytest
from PIL import Image

from core.config.loader import load_app_config
from core.exceptions.vision import ValidationError
from vision.validation.image_validator import ImageValidator


def test_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "sample.gif"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(path, format="GIF")
    validator = ImageValidator(load_app_config())
    with pytest.raises(ValidationError):
        validator.validate(path)


def test_rejects_too_small_image(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(path)
    validator = ImageValidator(load_app_config())
    with pytest.raises(ValidationError):
        validator.validate(path)


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    validator = ImageValidator(load_app_config())
    with pytest.raises(ValidationError):
        validator.validate(path)
