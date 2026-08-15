"""Gamma correction and CLAHE-style local contrast."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gamma_correction(pixels: NDArray[np.uint8], gamma: float = 1.1) -> NDArray[np.uint8]:
    """Apply gamma correction."""
    normalized = pixels.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1.0 / max(gamma, 0.1))
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_clahe(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply luminance-only CLAHE — never per-channel RGB equalization (gray collapse)."""
    try:
        import cv2

        if pixels.ndim == 2:
            gray = pixels
            enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
            return np.asarray(enhanced, dtype=np.uint8)
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
        channel, a, b = cv2.split(lab)
        # Mild clipLimit preserves natural color while lifting local contrast.
        channel = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(channel)
        merged = cv2.merge((channel, a, b))
        converted = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return np.asarray(converted, dtype=np.uint8)
    except Exception:
        return _luminance_equalize(pixels)


def _luminance_equalize(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Equalize luminance only — per-channel RGB EQ was a gray-shift root cause."""
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return pixels
    working = pixels.astype(np.float32)
    luminance = 0.2126 * working[:, :, 0] + 0.7152 * working[:, :, 1] + 0.0722 * working[:, :, 2]
    hist, _ = np.histogram(luminance.flatten(), 256, (0, 256))
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / max(cdf.max() - cdf.min(), 1)
    equalized = np.interp(luminance.flatten(), np.arange(256), cdf).reshape(luminance.shape)
    scale = equalized / np.maximum(luminance, 1.0)
    scale = np.clip(scale, 0.85, 1.15)
    result = np.clip(working * scale[:, :, None], 0, 255).astype(np.uint8)
    return result
