"""Bundled runtime resources (translations, etc.)."""

from __future__ import annotations

from core.resources.translations_loader import (
    SUPPORTED_LANGUAGES,
    load_all_translation_catalogs,
    load_translation_catalog,
    translation_source_path,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "load_all_translation_catalogs",
    "load_translation_catalog",
    "translation_source_path",
]
