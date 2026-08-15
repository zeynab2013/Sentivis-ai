"""Exposure and low-light correction helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision.enhancement.luminance import mean_luminance


def correct_exposure(
    pixels: NDArray[np.uint8],
    *,
    target_brightness: float = 0.5,
    max_scale: float = 1.18,
    min_scale: float = 0.92,
) -> NDArray[np.uint8]:
    """Gently scale luminance toward a target — never crush well-exposed scenes.

    Uses Rec.601 luma (not RGB channel mean) and clamps gain so bright images
    cannot be force-darkened to mid-gray.
    """
    if pixels.size == 0:
        return pixels
    mean = mean_luminance(pixels)
    if mean <= 1e-4:
        return pixels
    # Skip when already close to target — preserve original exposure.
    if abs(mean - target_brightness) < 0.06:
        return pixels
    raw_scale = target_brightness / mean
    scale = float(np.clip(raw_scale, min_scale, max_scale))
    if abs(scale - 1.0) < 0.02:
        return pixels
    working = pixels.astype(np.float32) * scale
    return np.clip(working, 0, 255).astype(np.uint8)


def enhance_low_light(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Lift shadows for under-exposed images."""
    if pixels.size == 0:
        return pixels
    working = pixels.astype(np.float32) / 255.0
    lifted = np.power(working, 0.85)
    return np.clip(lifted * 255.0, 0, 255).astype(np.uint8)
