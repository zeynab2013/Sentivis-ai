"""Luminance / color-preservation helpers for enhancement accept/reject."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_luminance(pixels: NDArray[np.uint8]) -> float:
    """Rec.601 luma in [0, 1]."""
    if pixels.size == 0:
        return 0.0
    arr = pixels.astype(np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(luma.mean() / 255.0)


def median_luminance(pixels: NDArray[np.uint8]) -> float:
    if pixels.size == 0:
        return 0.0
    arr = pixels.astype(np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.median(luma) / 255.0)


def mean_chroma(pixels: NDArray[np.uint8]) -> float:
    """Mean channel std as a simple chroma proxy in [0, ~180]."""
    if pixels.size == 0:
        return 0.0
    return float(np.std(pixels.astype(np.float32), axis=2).mean())


def brightness_collapsed(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    *,
    max_drop: float = 0.04,
    protect_above: float = 0.32,
) -> bool:
    """True when enhancement darkens a reasonably exposed original too much."""
    before = mean_luminance(original)
    after = mean_luminance(enhanced)
    if before < protect_above:
        # Underexposed originals may legitimately brighten; only reject huge drops.
        return after + 0.08 < before
    return after + max_drop < before


def color_shifted(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    *,
    max_channel_delta: float = 18.0,
) -> bool:
    """True when mean RGB shifts excessively versus the original."""
    if original.shape != enhanced.shape:
        # Upscaled images: compare resized stats on a shared scale.
        try:
            from PIL import Image

            o = np.asarray(
                Image.fromarray(original).resize(
                    (enhanced.shape[1], enhanced.shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
        except Exception:  # noqa: BLE001
            return False
    else:
        o = original.astype(np.float32)
    e = enhanced.astype(np.float32)
    delta = np.abs(o.mean(axis=(0, 1)) - e.mean(axis=(0, 1)))
    return bool(float(delta.max()) > max_channel_delta)
