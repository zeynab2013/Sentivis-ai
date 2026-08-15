"""Synthetic image fixtures for acceptance tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_standard_image(path: Path, *, width: int = 128, height: int = 128) -> Path:
    """Create a valid PNG suitable for pipeline analysis."""
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    row = np.tile(gradient, (height, 1))
    rgb = np.stack([row, row[:, ::-1], np.full((height, width), 128, dtype=np.uint8)], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def create_large_image(path: Path, *, width: int = 2048, height: int = 1536) -> Path:
    """Create a large but valid PNG within configured limits."""
    return create_standard_image(path, width=width, height=height)


def create_corrupted_image(path: Path) -> Path:
    """Create a file with PNG header but invalid payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return path


def create_empty_file(path: Path) -> Path:
    """Create a zero-byte file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path
