"""Unit tests for cache maintenance."""

from dataclasses import replace
from pathlib import Path

from core.config.loader import load_app_config
from services.cache.cache_manager import CacheManager
from services.runtime.cache_maintenance import CacheMaintenanceService


def test_cache_maintenance_detects_orphans(tmp_path: Path) -> None:
    app_config = load_app_config()
    config = replace(
        app_config,
        paths=replace(app_config.paths, cache_dir=tmp_path, model_search_paths=()),
    )
    cache = CacheManager(config)
    orphan = tmp_path / "stale.bin"
    orphan.write_bytes(b"orphan")
    maintenance = CacheMaintenanceService(config, cache)
    report = maintenance.report_size()
    assert "stale.bin" in report.orphaned_files


def test_cache_maintenance_safe_cleanup_removes_orphans(tmp_path: Path) -> None:
    app_config = load_app_config()
    config = replace(
        app_config,
        paths=replace(app_config.paths, cache_dir=tmp_path, model_search_paths=()),
    )
    cache = CacheManager(config)
    orphan = tmp_path / "stale.bin"
    orphan.write_bytes(b"orphan")
    maintenance = CacheMaintenanceService(config, cache)
    cleanup = maintenance.safe_cleanup()
    assert cleanup.removed_orphans == 1
    assert not orphan.exists()
