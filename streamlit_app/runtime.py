"""Portable runtime bootstrap for Streamlit launches."""

from __future__ import annotations

from pathlib import Path

from core.utils.paths import ensure_project_root_on_path, ensure_runtime_directories


def configure_runtime() -> Path:
    """Prepare import path and runtime directories before backend imports."""
    root = ensure_project_root_on_path()
    ensure_runtime_directories(root)
    return root
