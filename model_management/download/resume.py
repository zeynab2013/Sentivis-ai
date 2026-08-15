"""Resumable partial download helpers."""

from __future__ import annotations

from pathlib import Path


def partial_path(destination: Path) -> Path:
    """Return the temporary partial download path."""
    return destination.with_suffix(destination.suffix + ".partial")


def resume_offset(partial: Path) -> int:
    """Return bytes already downloaded for resume."""
    if partial.is_file():
        return partial.stat().st_size
    return 0


def finalize_partial(partial: Path, destination: Path) -> None:
    """Move completed partial file to final destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    partial.replace(destination)
