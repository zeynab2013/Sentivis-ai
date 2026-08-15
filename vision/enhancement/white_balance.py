"""Automatic white balance correction."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def white_balance(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Neutral-cast white balance only — never gray-world a chromatic scene.

    Classic gray-world equalizes channel means. On colorful images (red jackets,
    brown horses, green fields) that destroys hue and feeds the color estimator
    a near-achromatic mean → systematic gray predictions.
    """
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return pixels
    working = pixels.astype(np.float32)
    means = working.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    if gray <= 1.0:
        return pixels
    mx = float(means.max())
    mn = float(means.min())
    # Chromatic scene / colored illumination: skip — preservation over "correction".
    if mx > 1.0 and (mx - mn) / mx > 0.10:
        return pixels
    scale = gray / np.maximum(means, 1.0)
    # Tiny gains only for near-neutral casts.
    scale = np.clip(scale, 0.92, 1.08)
    balanced = np.clip(working * scale, 0, 255).astype(np.uint8)
    return cast(NDArray[np.uint8], balanced)
