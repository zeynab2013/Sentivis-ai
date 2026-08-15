"""UI preferences persisted locally (no DI changes)."""

from __future__ import annotations

from PySide6.QtCore import QSettings

_ORG = "SentivisAI"
_APP = "Desktop"


def load_language() -> str:
    return str(QSettings(_ORG, _APP).value("language", "en"))


def save_language(code: str) -> None:
    QSettings(_ORG, _APP).setValue("language", code)


def load_competition_mode() -> bool:
    return bool(QSettings(_ORG, _APP).value("competition_mode", False, type=bool))


def save_competition_mode(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("competition_mode", enabled)


def load_high_contrast() -> bool:
    return bool(QSettings(_ORG, _APP).value("high_contrast", False, type=bool))


def save_high_contrast(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("high_contrast", enabled)


def load_large_font() -> bool:
    return bool(QSettings(_ORG, _APP).value("large_font", False, type=bool))


def save_large_font(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("large_font", enabled)


def load_enable_enhancement() -> bool:
    return bool(QSettings(_ORG, _APP).value("enable_enhancement", True, type=bool))


def save_enable_enhancement(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("enable_enhancement", enabled)


def load_enable_super_resolution() -> bool:
    return bool(QSettings(_ORG, _APP).value("enable_super_resolution", False, type=bool))


def save_enable_super_resolution(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("enable_super_resolution", enabled)


def load_enable_sam2() -> bool:
    return bool(QSettings(_ORG, _APP).value("enable_sam2", True, type=bool))


def save_enable_sam2(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("enable_sam2", enabled)


def load_comparison_mode() -> bool:
    return bool(QSettings(_ORG, _APP).value("comparison_mode", False, type=bool))


def save_comparison_mode(enabled: bool) -> None:
    QSettings(_ORG, _APP).setValue("comparison_mode", enabled)
