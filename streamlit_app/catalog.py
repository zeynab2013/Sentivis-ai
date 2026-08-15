"""Qt-free translation catalog loader for Streamlit."""

from __future__ import annotations

from core.resources import SUPPORTED_LANGUAGES, load_all_translation_catalogs

_DEFAULT_LANGUAGE = "en"


class TranslationCatalog:
    """Loads JSON translation catalogs without PySide6."""

    def __init__(self) -> None:
        self._language = _DEFAULT_LANGUAGE
        self._catalogs: dict[str, dict[str, str]] = {}
        self._load_all()

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, code: str) -> None:
        normalized = code.lower().strip()
        if normalized not in SUPPORTED_LANGUAGES:
            normalized = _DEFAULT_LANGUAGE
        self._language = normalized

    def translate(self, key: str, **params: object) -> str:
        catalog = self._catalogs.get(self._language, {})
        fallback = self._catalogs.get(_DEFAULT_LANGUAGE, {})
        text = catalog.get(key) or fallback.get(key)
        if text is None:
            # Streamlit reruns / editable installs can leave a stale empty catalog.
            self._load_all()
            catalog = self._catalogs.get(self._language, {})
            fallback = self._catalogs.get(_DEFAULT_LANGUAGE, {})
            text = catalog.get(key) or fallback.get(key) or key
        if params:
            try:
                return text.format(**params)
            except (KeyError, ValueError):
                return text
        return text

    def _load_all(self) -> None:
        self._catalogs = load_all_translation_catalogs()


_catalog = TranslationCatalog()


def get_catalog() -> TranslationCatalog:
    return _catalog
