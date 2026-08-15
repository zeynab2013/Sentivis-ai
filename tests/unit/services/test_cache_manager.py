"""Unit tests for cache corruption handling."""

from dataclasses import replace
from pathlib import Path

from core.config.loader import load_app_config
from services.cache.cache_manager import CacheManager


def test_read_json_returns_none_for_corrupt_cache(tmp_path: Path) -> None:
    app_config = load_app_config()
    config = replace(app_config, paths=replace(app_config.paths, cache_dir=tmp_path))
    cache = CacheManager(config)
    corrupt = tmp_path / "broken.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    assert cache.read_json("broken") is None
    assert not corrupt.exists()


def test_read_json_returns_payload_for_valid_cache(tmp_path: Path) -> None:
    app_config = load_app_config()
    config = replace(app_config, paths=replace(app_config.paths, cache_dir=tmp_path))
    cache = CacheManager(config)
    cache.store_json("valid", {"ok": True})
    assert cache.read_json("valid") == {"ok": True}
