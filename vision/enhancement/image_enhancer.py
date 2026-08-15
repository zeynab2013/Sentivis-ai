"""High-level image enhancer facade."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.config.app_config import AppConfig
from core.contracts.image_quality import ImageQualityReport
from vision.enhancement.enhancement_pipeline import EnhancementPipeline


class ImageEnhancer:
    """Facade for adaptive image enhancement."""

    def __init__(self, app_config: AppConfig) -> None:
        self._pipeline = EnhancementPipeline(
            app_config.image.enhancement,
            app_config.paths.models_dir,
        )

    def enhance(
        self,
        pixels: NDArray[np.uint8],
        *,
        competition_mode: bool = False,
        enable_super_resolution: bool = False,
        enabled: bool = True,
    ) -> tuple[NDArray[np.uint8], ImageQualityReport]:
        """Enhance pixels when enabled and quality warrants it."""
        if not enabled:
            from vision.enhancement.quality_estimator import classify_quality, measure_quality

            metrics = measure_quality(pixels)
            level = classify_quality(metrics)
            already_clear = level == "HIGH"
            return pixels, ImageQualityReport(
                metrics=metrics,
                enhancement_operations=(),
                enhancement_applied=False,
                processing_time_ms=0.0,
                improvement_percent=0.0,
                before_quality=metrics.estimated_quality,
                after_quality=metrics.estimated_quality,
                quality_level=level,
                enhancement_attempted=False,
                enhancement_verified=False,
                # Do not label disabled MEDIUM/LOW as "Already clear".
                enhancement_status=(
                    "ENHANCEMENT_NOT_REQUIRED" if already_clear else "ENHANCEMENT_FAILED"
                ),
                verification_reason=(
                    "Enhancement skipped: input quality already sufficient."
                    if already_clear
                    else "Enhancement skipped: disabled in settings."
                ),
            )
        return self._pipeline.process(
            pixels,
            competition_mode=competition_mode,
            enable_super_resolution=enable_super_resolution,
        )
