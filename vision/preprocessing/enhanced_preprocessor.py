"""Preprocessor with adaptive image enhancement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from core.config.app_config import AppConfig
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.pipeline import AnalysisOptions
from core.logging import get_logger
from vision.enhancement.image_enhancer import ImageEnhancer
from vision.preprocessing.standard_preprocessor import StandardPreprocessor

logger = get_logger(__name__)


class EnhancedPreprocessor:
    """Wraps standard preprocessing with optional quality-aware enhancement."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base = StandardPreprocessor(config)
        self._enhancer = ImageEnhancer(config)
        self._last_options = AnalysisOptions()

    def set_analysis_options(self, options: AnalysisOptions) -> None:
        """Set per-run analysis options for enhancement behavior."""
        self._last_options = options

    def preprocess(self, image: ValidatedImage) -> PreprocessedImage:
        """Enhance when warranted, then produce inference buffers."""
        original_pixels = image.pixels
        enhanced_pixels, quality_report = self._enhancer.enhance(
            original_pixels,
            competition_mode=self._last_options.competition_mode,
            enable_super_resolution=self._last_options.enable_super_resolution,
            enabled=self._last_options.enable_enhancement and self._config.image.enhancement.enabled,
        )
        enhanced_source = ValidatedImage(
            path=image.path,
            width=enhanced_pixels.shape[1],
            height=enhanced_pixels.shape[0],
            format_name=image.format_name,
            size_bytes=image.size_bytes,
            pixels=enhanced_pixels,
        )
        base = self._base.preprocess(enhanced_source)
        logger.debug(
            "Enhancement applied=%s operations=%s",
            quality_report.enhancement_applied,
            quality_report.enhancement_operations,
        )
        return PreprocessedImage(
            source=enhanced_source,
            display_pixels=enhanced_pixels,
            inference_pixels=base.inference_pixels,
            inference_width=base.inference_width,
            inference_height=base.inference_height,
            # Always retain pristine pixels for color/clothing — never sample enhanced output.
            original_display_pixels=original_pixels,
            quality_report=quality_report,
            enhancement_applied=quality_report.enhancement_applied,
        )

    @staticmethod
    def resize_inference_buffer(pixels: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
        """Utility for tests — resize display pixels to inference size."""
        pil_image = Image.fromarray(pixels)
        resized = pil_image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)
