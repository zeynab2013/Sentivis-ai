"""Streamlit UI preferences (JSON persistence, no Qt)."""

from __future__ import annotations

import json

from core.utils.paths import project_root

_PREFS_PATH = project_root() / ".sentivis" / "streamlit_prefs.json"


def _load_raw() -> dict[str, object]:
    if not _PREFS_PATH.is_file():
        return {}
    try:
        payload = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_raw(data: dict[str, object]) -> None:
    _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_bool(key: str, default: bool) -> bool:
    value = _load_raw().get(key, default)
    return bool(value)


def _set_bool(key: str, value: bool) -> None:
    data = _load_raw()
    data[key] = value
    _save_raw(data)


def load_language() -> str:
    return str(_load_raw().get("language", "en"))


def save_language(code: str) -> None:
    data = _load_raw()
    data["language"] = code
    _save_raw(data)


def load_competition_mode() -> bool:
    return _get_bool("competition_mode", False)


def save_competition_mode(enabled: bool) -> None:
    _set_bool("competition_mode", enabled)


def load_enable_enhancement() -> bool:
    return _get_bool("enable_enhancement", True)


def save_enable_enhancement(enabled: bool) -> None:
    _set_bool("enable_enhancement", enabled)


def load_enable_super_resolution() -> bool:
    return _get_bool("enable_super_resolution", True)


def save_enable_super_resolution(enabled: bool) -> None:
    _set_bool("enable_super_resolution", enabled)


def load_enable_sam2() -> bool:
    return _get_bool("enable_sam2", True)


def save_enable_sam2(enabled: bool) -> None:
    _set_bool("enable_sam2", enabled)


def load_comparison_mode() -> bool:
    return _get_bool("comparison_mode", False)


def save_comparison_mode(enabled: bool) -> None:
    _set_bool("comparison_mode", enabled)


def load_high_contrast() -> bool:
    return _get_bool("high_contrast", False)


def save_high_contrast(enabled: bool) -> None:
    _set_bool("high_contrast", enabled)
