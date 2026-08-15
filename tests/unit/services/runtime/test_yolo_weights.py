"""Tests for YOLO weight path resolution."""

from __future__ import annotations

from pathlib import Path

from core.config.loader import load_model_config
from core.utils.paths import normalize_optional_path
from services.runtime.yolo_weights import resolve_yolo_weights_path


def test_empty_weights_path_normalizes_to_none() -> None:
    assert normalize_optional_path("") is None
    assert normalize_optional_path(".") is None
    assert normalize_optional_path("  ") is None


def test_model_config_empty_weights_path_is_not_current_directory() -> None:
    model_config = load_model_config()
    assert model_config.yolo.weights_path is None


def test_resolve_yolo_weights_from_models_directory(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weights = models_dir / "yolo11x.pt"
    weights.write_bytes(b"x" * 2048)

    resolved = resolve_yolo_weights_path(
        variant="yolo11x",
        configured_path=None,
        search_paths=(models_dir,),
    )
    assert resolved == weights.resolve()


def test_resolve_prefers_configured_absolute_path(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    configured = models_dir / "custom_yolo11x.pt"
    configured.write_bytes(b"x" * 2048)

    resolved = resolve_yolo_weights_path(
        variant="yolo11x",
        configured_path=configured,
        search_paths=(models_dir,),
    )
    assert resolved == configured.resolve()
