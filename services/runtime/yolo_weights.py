"""YOLO weight path resolution across configuration and discovery."""

from __future__ import annotations

from pathlib import Path

from core.utils.paths import normalize_optional_path
from services.runtime.model_discovery import ModelDiscoveryResult, discover_model_files


def resolve_yolo_weights_path(
    *,
    variant: str,
    configured_path: Path | None,
    search_paths: tuple[Path, ...],
    discovery: ModelDiscoveryResult | None = None,
) -> Path | None:
    """Resolve an existing YOLO weights file from config and configured search paths."""
    normalized = normalize_optional_path(configured_path)
    if normalized is not None:
        for candidate in _expand_candidates(normalized, search_paths):
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved

    result = discovery if discovery is not None else discover_model_files(search_paths)
    variant_key = variant.strip().lower()

    for item in result.weight_files:
        if variant_key and item.path.name.lower().startswith(variant_key) and item.path.is_file():
            return item.path.resolve()

    for item in result.weight_files:
        if item.path.is_file():
            return item.path.resolve()

    for root in search_paths:
        for name in (f"{variant.strip()}.pt", "yolo11x.pt"):
            candidate = (root / name).resolve()
            if candidate.is_file():
                return candidate

    return None


def _expand_candidates(path: Path, search_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if path.is_absolute():
        return (path,)
    return tuple(dict.fromkeys((path, *(root / path for root in search_paths))))
