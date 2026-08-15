"""Branding asset resolution for Streamlit (Qt-free)."""

from __future__ import annotations

from pathlib import Path

from core.utils.paths import project_root, resource_path


def logo_path() -> Path:
    """Return logo.png, logo.svg, or bundled fallback."""
    for parts in (
        ("branding", "logo", "logo.png"),
        ("branding", "logo", "logo.svg"),
        ("icons", "app_icon.svg"),
    ):
        path = resource_path(*parts)
        if path.is_file():
            return path
    return project_root() / "assets" / "icons" / "app_icon.svg"


def favicon_path() -> Path:
    """Browser favicon — prefer PNG logo."""
    png = resource_path("branding", "logo", "logo.png")
    if png.is_file():
        return png
    return logo_path()


def splash_path() -> Path:
    """Optional splash asset."""
    for name in ("splash.png", "splash.svg"):
        path = resource_path("branding", "splash", name)
        if path.is_file():
            return path
    return logo_path()
