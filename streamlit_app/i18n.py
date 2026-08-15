"""Streamlit i18n bridge (Qt-free)."""

from __future__ import annotations

import os

from streamlit_app import preferences as prefs
from streamlit_app.catalog import SUPPORTED_LANGUAGES, get_catalog

LANGUAGE_LABELS = {
    "en": "English",
    "fa": "فارسی",
    "de": "Deutsch",
    "es": "Español",
    "zh": "中文",
}


def _publish_language(code: str) -> None:
    """Keep caption/report language in lockstep with the visible Streamlit UI."""
    normalized = (code or "en").lower().strip()
    if normalized not in SUPPORTED_LANGUAGES:
        normalized = "en"
    os.environ["SENTIVIS_UI_LANGUAGE"] = normalized
    get_catalog().set_language(normalized)

    # Register domain-safe provider so language/ never imports streamlit_app.
    def _provider() -> str:
        return get_catalog().language

    try:
        from core.config.ui_language import register_ui_language_provider

        register_ui_language_provider(_provider)
    except Exception:  # noqa: BLE001
        pass

    try:
        from language.refinement.caption_refiner import clear_ui_language_cache
        from ui.i18n.translator import get_translator
        from ui.preferences.ui_preferences import save_language as save_desktop_language

        clear_ui_language_cache()
        save_desktop_language(normalized)
        get_translator().set_language(normalized)
    except Exception:  # noqa: BLE001
        pass


def sync_language_from_prefs() -> None:
    _publish_language(prefs.load_language())


def set_language(code: str) -> None:
    prefs.save_language(code)
    _publish_language(code)


def t(key: str, **params: object) -> str:
    return get_catalog().translate(key, **params)


__all__ = ["LANGUAGE_LABELS", "SUPPORTED_LANGUAGES", "set_language", "sync_language_from_prefs", "t"]
