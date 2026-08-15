"""Smoke tests verifying core application subsystems start correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.container import DependencyContainer
from app.startup.orchestrator import StartupOrchestrator
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from services.runtime.assets import build_runtime_assets


@pytest.mark.acceptance
def test_application_startup_completes_all_stages() -> None:
    result = StartupOrchestrator().run()
    assert result.context.facade is not None
    assert len(result.report.stages) == 8


@pytest.mark.acceptance
def test_dependency_container_creation() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    assert context.facade is not None
    assert context.main_controller is not None
    assert context.runtime_status.health_score >= 0


@pytest.mark.acceptance
def test_configuration_loading() -> None:
    result = StartupOrchestrator().run()
    settings = result.settings
    assert settings.app_config.app_name
    assert settings.model_config is not None
    assert settings.theme_config is not None
    assert settings.analysis_config is not None


@pytest.mark.acceptance
def test_plugin_loading() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    plugin_ids = tuple(plugin.identifier for plugin in context.plugin_registry.list_plugins())
    assert "vision.yolo_v8n" in plugin_ids
    assert "language.blip_base" in plugin_ids
    assert "language.gemma_2b" in plugin_ids


@pytest.mark.acceptance
def test_model_discovery_registers_three_models() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    assert len(context.model_registry.records) == 3


@pytest.mark.acceptance
def test_asset_loading_inventory() -> None:
    app_config = load_app_config()
    assets = build_runtime_assets(app_config)
    inventories = assets.inventory()
    assert len(inventories) == 9
    icon_inventory = next(item for item in inventories if item.category.value == "icons")
    assert icon_inventory.file_count >= 1


@pytest.mark.acceptance
def test_startup_diagnostics_export(tmp_path: Path) -> None:
    result = StartupOrchestrator().run()
    json_path, text_path = result.diagnostics.write(tmp_path)
    assert json_path.is_file()
    assert text_path.is_file()
    assert "Sentivis AI" in text_path.read_text(encoding="utf-8")


@pytest.mark.acceptance
def test_graceful_shutdown() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    context.model_manager.release_all()
    context.memory_manager.clear_gpu_cache()
