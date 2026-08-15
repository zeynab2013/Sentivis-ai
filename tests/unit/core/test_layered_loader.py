"""Unit tests for layered configuration loading."""

from core.config.layered_loader import deep_merge, load_layered_toml
from core.utils.paths import project_root


def test_deep_merge_overrides_nested_values() -> None:
    base = {"logging": {"level": "INFO", "console_enabled": True}, "app": {"name": "A"}}
    override = {"logging": {"level": "DEBUG"}}
    merged = deep_merge(base, override)
    assert merged["logging"]["level"] == "DEBUG"
    assert merged["logging"]["console_enabled"] is True
    assert merged["app"]["name"] == "A"


def test_load_layered_toml_reads_default_config() -> None:
    data = load_layered_toml(project_root() / "config" / "app.default.toml")
    assert "app" in data
    assert "logging" in data
