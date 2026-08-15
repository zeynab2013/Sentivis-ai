"""Translation resource loading — source checkout and package data."""

from __future__ import annotations

from core.resources import (
    SUPPORTED_LANGUAGES,
    load_all_translation_catalogs,
    load_translation_catalog,
    translation_source_path,
)


def test_english_catalog_loads_from_package_or_checkout() -> None:
    catalog = load_translation_catalog("en")
    assert catalog
    assert any(k.startswith("streamlit.") or k.startswith("app.") or "." in k for k in catalog)


def test_all_supported_languages_load() -> None:
    catalogs = load_all_translation_catalogs()
    for code in SUPPORTED_LANGUAGES:
        assert code in catalogs
        assert catalogs[code]


def test_translation_source_path_resolves() -> None:
    path = translation_source_path("en")
    assert path is not None
    assert path.is_file()
    assert path.name == "en.json"


def test_unknown_language_returns_empty() -> None:
    assert load_translation_catalog("xx") == {}
