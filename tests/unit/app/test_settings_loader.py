"""Unit tests for settings loader."""

from app.settings_loader import load_application_settings


def test_load_application_settings_returns_all_configs() -> None:
    settings = load_application_settings()
    assert settings.app_config.app_name == "Sentivis AI"
    assert settings.model_config.yolo.variant
    assert settings.analysis_config.attributes.size_small_max_ratio > 0
    assert settings.theme_config.name
    assert settings.sources.sources
