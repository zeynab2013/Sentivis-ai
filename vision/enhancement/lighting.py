"""Adaptive brightness and contrast adjustments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance


def adaptive_brightness(pixels: NDArray[np.uint8], target: float = 0.5) -> NDArray[np.uint8]:
    """Gently shift brightness toward target mean luminance.

    Gains are clamped so well-exposed images are never crushed toward mid-gray.
    """
    gray = pixels.astype(np.float32)
    if gray.ndim == 3:
        current = float(
            np.mean(0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]) / 255.0
        )
    else:
        current = float(np.mean(gray) / 255.0)
    if abs(current - target) < 0.06:
        return pixels
    # Preserve original exposure: only mild lifts/cuts.
    factor = max(0.92, min(1.25, target / max(current, 0.05)))
    if abs(factor - 1.0) < 0.03:
        return pixels
    image = Image.fromarray(pixels)
    return np.asarray(ImageEnhance.Brightness(image).enhance(factor), dtype=np.uint8)


def enhance_contrast(pixels: NDArray[np.uint8], factor: float = 1.15) -> NDArray[np.uint8]:
    """Boost contrast using PIL."""
    image = Image.fromarray(pixels)
    return np.asarray(ImageEnhance.Contrast(image).enhance(factor), dtype=np.uint8)
