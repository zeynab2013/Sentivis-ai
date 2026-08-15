"""Layered configuration merge utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base recursively."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_layered_toml(default_path: Path, user_path: Path | None = None) -> dict[str, object]:
    """Load default TOML and merge user override when present."""
    from core.config.loader import _read_toml

    data = _read_toml(default_path)
    if user_path is not None and user_path.is_file():
        user_data = _read_toml(user_path)
        if isinstance(user_data, dict):
            data = deep_merge(data, cast(dict[str, Any], user_data))
    return data
