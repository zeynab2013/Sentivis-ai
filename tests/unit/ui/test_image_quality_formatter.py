"""Tests for image quality result formatting."""

from __future__ import annotations

from dataclasses import replace

from core.contracts.image_quality import ImageQualityMetrics, ImageQualityReport
from tests.unit.services.test_export_reports import _sample_result
from ui.formatters.result_formatters import format_image_quality


def test_format_image_quality_without_report() -> None:
    assert "No image quality report" in format_image_quality(_sample_result())


def test_format_image_quality_with_report() -> None:
    metrics = ImageQualityMetrics(
        resolution_width=640,
        resolution_height=480,
        brightness=0.5,
        contrast=0.4,
        blur_score=0.2,
        noise_score=0.1,
        sharpness=0.7,
        dynamic_range=0.6,
        compression_artifact_score=0.05,
        estimated_quality=0.75,
    )
    report = ImageQualityReport(
        metrics=metrics,
        enhancement_operations=("clahe",),
        enhancement_applied=True,
        processing_time_ms=12.5,
        improvement_percent=8.0,
        before_quality=0.7,
        after_quality=0.78,
        quality_level="MEDIUM",
    )
    result = replace(_sample_result(), image_quality=report)
    text = format_image_quality(result)
    assert "640×480" in text
    assert "clahe" in text
    assert "Quality: MEDIUM" in text
    assert "Enhancement: Verified" in text
