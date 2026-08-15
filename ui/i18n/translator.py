"""Translation manager with hot-reload support."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from core.resources import SUPPORTED_LANGUAGES, load_all_translation_catalogs

_DEFAULT_LANGUAGE = "en"


class Translator(QObject):
    """Loads JSON translation catalogs and resolves UI keys."""

    language_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
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
        if normalized == self._language:
            return
        self._language = normalized
        self.language_changed.emit(normalized)

    def translate(self, key: str, **params: object) -> str:
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


_translator = Translator()


def get_translator() -> Translator:
    return _translator


def t(key: str, **params: object) -> str:
    return get_translator().translate(key, **params)


def tr(key: str, **params: object) -> str:
    """Desktop alias used throughout the Qt UI."""
    return t(key, **params)
