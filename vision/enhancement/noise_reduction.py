"""Noise reduction filters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reduce_noise(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Reduce noise using OpenCV when available, else mild blur.

    OpenCV denoise expects BGR — convert explicitly so RGB Sentivis pixels
    do not get a color/brightness cast.
    """
    try:
        import cv2

        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
        rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb, dtype=np.uint8)
    except Exception:
        from PIL import Image, ImageFilter

        return np.asarray(
            Image.fromarray(pixels).filter(ImageFilter.MedianFilter(size=3)),
            dtype=np.uint8,
        )
