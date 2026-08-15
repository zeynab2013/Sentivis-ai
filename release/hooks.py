"""Release engineering hooks for application bootstrap."""

from __future__ import annotations

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget

from release.about_dialog import show_about_dialog
from release.metadata import ReleaseInfo


def attach_release_hooks(window: QWidget, release_info: ReleaseInfo) -> None:
    """Attach release-engineering shortcuts without modifying frozen UI widgets."""
    QShortcut(
        QKeySequence("F1"),
        window,
        lambda: show_about_dialog(window, release_info),
    )
