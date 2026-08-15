"""Sharpening and JPEG artifact mitigation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter


def sharpen(pixels: NDArray[np.uint8], factor: float = 1.12) -> NDArray[np.uint8]:
    """Gentle unsharp mask — preserve edges without halos or synthetic texture."""
    image = Image.fromarray(pixels)
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=0.9, percent=105, threshold=4))
    # Keep the secondary sharpness boost mild.
    boost = min(max(factor, 1.0), 1.2)
    return np.asarray(ImageEnhance.Sharpness(sharpened).enhance(boost), dtype=np.uint8)


def reduce_jpeg_artifacts(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Mild smoothing to reduce blocky JPEG artifacts."""
    image = Image.fromarray(pixels)
    return np.asarray(image.filter(ImageFilter.SMOOTH_MORE), dtype=np.uint8)
