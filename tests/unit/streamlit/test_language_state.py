"""Language session-state contract for Streamlit UI switching."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit_app.catalog import SUPPORTED_LANGUAGES, TranslationCatalog
from streamlit_app.i18n import LANGUAGE_LABELS


ROOT = Path(__file__).resolve().parents[3]


def test_supported_languages_and_labels() -> None:
    assert SUPPORTED_LANGUAGES == ("en", "fa", "de", "es", "zh")
    for code in SUPPORTED_LANGUAGES:
        assert code in LANGUAGE_LABELS


def test_catalog_language_switch_changes_visible_strings() -> None:
    catalog = TranslationCatalog()
    catalog.set_language("en")
    english = catalog.translate("button.analyze")
    catalog.set_language("de")
    german = catalog.translate("button.analyze")
    assert english != german
    assert german  # non-empty


def test_main_defines_language_session_helpers() -> None:
    source = (ROOT / "streamlit_app" / "main.py").read_text(encoding="utf-8")
    assert "st.session_state.ui_language" in source
    assert "def _switch_language" in source
    assert "def _apply_caption_language" in source
    assert "_render_main_language_selector" in source
    assert "main_language_select" in source
    assert "st.session_state.display_image" in source


def test_ui_catalogs_cover_english_keys() -> None:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    for lang in SUPPORTED_LANGUAGES:
        data = json.loads((ROOT / "translations" / f"{lang}.json").read_text(encoding="utf-8"))
        missing = sorted(set(en) - set(data))
        assert not missing, f"{lang} missing keys: {missing[:20]}"
        assert len(data) == len(en)


def test_german_catalog_complete_count() -> None:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    de = json.loads((ROOT / "translations" / "de.json").read_text(encoding="utf-8"))
    assert len(en) == len(de)
    assert len(de) >= 470
