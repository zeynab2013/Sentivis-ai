"""Integration tests for dependency container."""

from app.container import DependencyContainer
from core.config.loader import (
    load_analysis_config,
    load_app_config,
    load_model_config,
    load_theme_config,
)


def test_dependency_container_builds() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    assert context.facade is not None
    assert context.model_manager is not None
    plugin_ids = {plugin.identifier for plugin in context.plugin_registry.list_plugins()}
    assert {"vision.yolo_v8n", "language.blip_base", "language.gemma_2b"} <= plugin_ids
    assert "language.florence2" in plugin_ids
