"""Image pipeline DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.contracts.image_quality import ImageQualityReport

@dataclass(frozen=True)
class ImagePayload:
    """Raw image reference before validation."""

    path: Path


@dataclass(frozen=True)
class ValidatedImage:
    """Image that passed validation checks."""

    path: Path
    width: int
    height: int
    format_name: str
    size_bytes: int
    pixels: NDArray[np.uint8]


@dataclass(frozen=True)
class PreprocessedImage:
    """Image prepared for model inference."""

    source: ValidatedImage
    display_pixels: NDArray[np.uint8]
    inference_pixels: NDArray[np.uint8]
    inference_width: int
    inference_height: int
    original_display_pixels: NDArray[np.uint8] | None = None
    quality_report: ImageQualityReport | None = None
    enhancement_applied: bool = False