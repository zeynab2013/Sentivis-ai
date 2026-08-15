"""UI language resolution without UI-framework imports.

Streamlit / Qt layers register a provider at startup. Domain code
(language/, analysis/) must call ``resolve_ui_language()`` instead of
importing ``streamlit_app`` or ``ui`` directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable

SUPPORTED_UI_LANGUAGES: tuple[str, ...] = ("en", "fa", "de", "es", "zh")
DEFAULT_UI_LANGUAGE = "en"

_UI_LANGUAGE_PROVIDER: Callable[[], str] | None = None


def register_ui_language_provider(provider: Callable[[], str] | None) -> None:
    """Register or clear the active UI language provider (Streamlit/Qt)."""
    global _UI_LANGUAGE_PROVIDER
    _UI_LANGUAGE_PROVIDER = provider


def clear_ui_language_provider() -> None:
    """Remove the registered provider (tests / shutdown)."""
    register_ui_language_provider(None)


def resolve_ui_language() -> str:
    """Return the active UI language code.

    Priority:
    1. ``SENTIVIS_UI_LANGUAGE`` process override
    2. Registered UI provider (Streamlit catalog / Qt prefs)
    3. Default English
    """
    override = (os.environ.get("SENTIVIS_UI_LANGUAGE") or "").lower().strip()
    if override in SUPPORTED_UI_LANGUAGES:
        return override

    provider = _UI_LANGUAGE_PROVIDER
    if provider is not None:
        try:
            code = (provider() or "").lower().strip()
            if code in SUPPORTED_UI_LANGUAGES:
                return code
        except Exception:  # noqa: BLE001 — provider must never break caption path
            pass

    return DEFAULT_UI_LANGUAGE
