"""Unit tests for startup model discovery."""

from app.startup.model_discovery import discover_models
from core.config.loader import load_model_config
from core.utils.paths import project_root


def test_discover_models_reports_all_kinds() -> None:
    model_config = load_model_config()
    report = discover_models(model_config, project_root() / "models")
    kinds = {entry.kind for entry in report.entries}
    assert kinds == {"yolo", "blip", "florence2", "gemma", "sam2", "realesrgan", "ollama"}
