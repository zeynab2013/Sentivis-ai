"""Unit tests for runtime asset managers."""

from core.config.loader import load_app_config
from services.runtime.assets import build_runtime_assets


def test_runtime_asset_managers_cover_all_categories() -> None:
    assets = build_runtime_assets(load_app_config())
    managers = assets.all_managers()
    categories = {manager.category.value for manager in managers}
    assert categories == {
        "models",
        "icons",
        "themes",
        "configuration",
        "samples",
        "export_templates",
        "logs",
        "cache",
        "temporary",
    }


def test_runtime_asset_inventory_reports_writable_paths() -> None:
    assets = build_runtime_assets(load_app_config())
    inventory = assets.inventory()
    assert len(inventory) == 9
    assert all(item.root for item in inventory)
