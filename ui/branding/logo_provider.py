"""Load branding logo from assets with safe fallbacks."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

from core.utils.paths import resource_path

_LOGO_MEMORY: dict[int, QPixmap] = {}


def branding_logo_path() -> Path:
    """Return preferred logo path (PNG) or bundled default SVG."""
    custom_png = resource_path("branding", "logo", "logo.png")
    if custom_png.is_file():
        return custom_png
    custom_svg = resource_path("branding", "logo", "logo.svg")
    if custom_svg.is_file():
        return custom_svg
    return resource_path("icons", "app_icon.svg")


def load_logo_pixmap(size: int = 128) -> QPixmap:
    """Load logo pixmap at requested size."""
    cached = _LOGO_MEMORY.get(size)
    if cached is not None and not cached.isNull():
        return cached

    path = branding_logo_path()
    pixmap = QPixmap()
    if path.suffix.lower() == ".svg":
        try:
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(str(path))
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
        except Exception:
            pixmap = QPixmap(str(path))
    else:
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
    if not pixmap.isNull():
        _LOGO_MEMORY[size] = pixmap
    return pixmap


def load_window_icon() -> QIcon:
    """Window/taskbar icon from branding assets."""
    pixmap = load_logo_pixmap(256)
    if pixmap.isNull():
        return QIcon()
    return QIcon(pixmap)
