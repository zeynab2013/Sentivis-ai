"""Image validation implementation."""

from pathlib import Path

import numpy as np
from PIL import Image

from core.config.app_config import AppConfig
from core.contracts.image import ValidatedImage
from core.exceptions.vision import ValidationError
from core.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
_MIN_DIMENSION = 32


class ImageValidator:
    """Validates image files before pipeline processing."""

    def __init__(self, config: AppConfig) -> None:
        self._max_dimension = config.image.max_dimension
        self._max_file_size = config.image.max_file_size_bytes

    def validate(self, path: Path) -> ValidatedImage:
        """Load and validate an image from disk."""
        if not path.exists():
            raise ValidationError(
                "The selected image could not be found.",
                f"Image path does not exist: {path}",
            )
        if not path.is_file():
            raise ValidationError(
                "The selected path is not a valid image file.",
                f"Path is not a file: {path}",
            )

        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise ValidationError(
                "The image file is empty.",
                f"Zero-byte file: {path}",
            )
        if size_bytes > self._max_file_size:
            raise ValidationError(
                "The image file is too large for analysis.",
                f"File size {size_bytes} exceeds limit {self._max_file_size}",
            )

        try:
            with Image.open(path) as handle:
                handle.verify()
            with Image.open(path) as handle:
                rgb_image = handle.convert("RGB")
                width, height = rgb_image.size
                format_name = handle.format or "UNKNOWN"
                pixels = np.asarray(rgb_image, dtype=np.uint8)
        except OSError as exc:
            raise ValidationError(
                "The image file appears to be corrupt or unsupported.",
                f"Failed to decode image {path}: {exc}",
            ) from exc

        if format_name not in _SUPPORTED_FORMATS:
            raise ValidationError(
                "This image format is not supported.",
                f"Unsupported format {format_name} for {path}",
            )

        if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
            raise ValidationError(
                "The image resolution is too small for analysis.",
                f"Dimensions {width}x{height} below minimum {_MIN_DIMENSION}px",
            )

        if width > self._max_dimension or height > self._max_dimension:
            raise ValidationError(
                "The image dimensions exceed the maximum allowed size.",
                f"Dimensions {width}x{height} exceed max {self._max_dimension}",
            )

        if pixels.size == 0:
            raise ValidationError(
                "The image contains no readable pixel data.",
                f"Empty pixel buffer for {path}",
            )

        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise ValidationError(
                "The image color data is not supported.",
                f"Unexpected color channels: {getattr(pixels, 'shape', None)}",
            )

        logger.info("Validated image %s (%dx%d)", path.name, width, height)
        return ValidatedImage(
            path=path,
            width=width,
            height=height,
            format_name=format_name,
            size_bytes=size_bytes,
            pixels=pixels,
        )
