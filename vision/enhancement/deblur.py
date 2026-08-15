"""Deblurring via OpenCV with safe fallback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def deblur(pixels: NDArray[np.uint8], *, strength: float = 1.18) -> NDArray[np.uint8]:
    """Mild unsharp deblur — avoid aggressive halos."""
    if pixels.size == 0:
        return pixels
    try:
        import cv2

        strength = float(np.clip(strength, 1.05, 1.28))
        blurred = cv2.GaussianBlur(pixels, (0, 0), sigmaX=0.9)
        sharpened = cv2.addWeighted(pixels, strength, blurred, 1.0 - strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:
        return pixels
