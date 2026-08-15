"""Installation-safe translation catalog loading.

Resolution order (first hit wins):
1. ``importlib.resources`` package data under ``core.resources.translations``
2. ``project_root()/translations/*.json`` (editable checkout / release tree)
3. ``SENTIVIS_PROJECT_ROOT/translations`` via project_root()
"""

from __future__ import annotations

import json
from pathlib import Path

from core.utils.paths import project_root

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "fa", "de", "es", "zh")


def translation_source_path(language: str) -> Path | None:
    """Return a readable filesystem path for ``language`` if one exists."""
    code = (language or "").lower().strip()
    if code not in SUPPORTED_LANGUAGES:
        return None

    # Package data (wheel / installed distribution).
    try:
        from importlib.resources import as_file, files

        resource = files("core.resources.translations").joinpath(f"{code}.json")
        with as_file(resource) as path:
            if path.is_file():
                return Path(path)
    except (FileNotFoundError, ModuleNotFoundError, TypeError, AttributeError, OSError):
        pass

    # Editable checkout / release folder layout.
    candidate = project_root() / "translations" / f"{code}.json"
    if candidate.is_file():
        return candidate
    return None


def load_translation_catalog(language: str) -> dict[str, str]:
    """Load one language catalog; empty dict when unavailable."""
    code = (language or "").lower().strip()
    if code not in SUPPORTED_LANGUAGES:
        return {}

    # Prefer importlib.resources text API (works for zip/wheel without extract).
    try:
        from importlib.resources import files

        resource = files("core.resources.translations").joinpath(f"{code}.json")
        text = resource.read_text(encoding="utf-8")
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    except (FileNotFoundError, ModuleNotFoundError, TypeError, AttributeError, OSError, json.JSONDecodeError):
        pass

    path = project_root() / "translations" / f"{code}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def load_all_translation_catalogs() -> dict[str, dict[str, str]]:
    """Load every supported language catalog that is available."""
    catalogs: dict[str, dict[str, str]] = {}
    for code in SUPPORTED_LANGUAGES:
        catalog = load_translation_catalog(code)
        if catalog:
            catalogs[code] = catalog
    return catalogs
