"""Integration tests for release wiring."""

from app.container import DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config


def test_application_context_includes_release_info() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    assert context.release_info.application_version == "1.0.0"
    assert "build" in context.facade.settings_view_model.app_version
