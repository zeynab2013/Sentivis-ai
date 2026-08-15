"""Unit tests for runtime model discovery."""

from pathlib import Path

from services.runtime.model_discovery import discover_model_files, resolve_model_search_paths


def test_resolve_model_search_paths_deduplicates(tmp_path: Path) -> None:
    primary = tmp_path / "models"
    primary.mkdir()
    paths = resolve_model_search_paths(primary, (primary,))
    assert paths == (primary.resolve(),)


def test_discover_model_files_finds_weights(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    weights = models / "yolov8n.pt"
    weights.write_bytes(b"fake-weights")
    result = discover_model_files((models,))
    assert len(result.weight_files) == 1
    assert result.weight_files[0].path.name == "yolov8n.pt"
