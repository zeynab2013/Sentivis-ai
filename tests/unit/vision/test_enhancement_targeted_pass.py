"""Targeted enhancement execution / honesty regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from vision.enhancement.enhancement_gate import compare_metrics
from vision.enhancement.enhancement_pipeline import EnhancementPipeline
from vision.enhancement.image_enhancer import ImageEnhancer
from vision.enhancement.quality_estimator import classify_quality, measure_quality
from core.config.app_config import AppConfig
from core.config.loader import load_app_config

_NO_SR = replace(
    DEFAULT_ENHANCEMENT_CONFIG,
    enable_super_resolution=False,
    sr_allow_download=False,
)


def _blurry_low(h: int = 180, w: int = 240) -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.integers(20, 90, size=(h, w, 3), dtype=np.uint8)
    soft = base.copy()
    for _ in range(4):
        soft[1:, 1:] = (
            (soft[1:, 1:].astype(np.uint16) + soft[:-1, :-1].astype(np.uint16)) // 2
        ).astype(np.uint8)
    return soft


def _soft_medium(h: int = 500, w: int = 700) -> np.ndarray:
    """Soft low/medium synthetic photo (not a hard checkerboard)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (80 + 60 * np.sin(xx / 40) + 40 * np.cos(yy / 35)).clip(0, 255)
    g = (90 + 50 * np.cos(xx / 55) + 30 * np.sin(yy / 28)).clip(0, 255)
    b = (70 + 45 * np.sin((xx + yy) / 50)).clip(0, 255)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    soft = img.copy()
    for _ in range(5):
        soft[1:, 1:] = (
            (soft[1:, 1:].astype(np.uint16) + soft[:-1, :-1].astype(np.uint16)) // 2
        ).astype(np.uint8)
    return soft


def _high_quality(h: int = 640, w: int = 800) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pixels = ((xx + yy) % 32 < 16).astype(np.uint8) * 200
    return np.stack([pixels, pixels, pixels], axis=-1)


def test_high_quality_not_unnecessarily_enhanced(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _high_quality()
    enhanced, report = pipeline.process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    assert report.quality_level == "HIGH"
    assert report.enhancement_attempted is False
    assert report.enhancement_applied is False
    assert report.enhancement_status == "ENHANCEMENT_NOT_REQUIRED"
    assert np.array_equal(pixels, enhanced)


def test_low_quality_attempts_and_status_matches(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _blurry_low()
    enhanced, report = pipeline.process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    assert report.quality_level in {"LOW", "MEDIUM"}
    assert report.enhancement_attempted is True
    if report.enhancement_applied:
        assert report.enhancement_verified is True
        assert report.enhancement_status in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}
        assert not np.array_equal(pixels, enhanced) or enhanced.shape != pixels.shape
    else:
        assert report.enhancement_verified is False
        assert np.array_equal(pixels, enhanced)
        assert report.enhancement_status in {
            "ENHANCEMENT_REJECTED",
            "ENHANCEMENT_ATTEMPTED_UNVERIFIED",
            "ENHANCEMENT_FAILED",
        }


def test_medium_soft_image_attempts_enhancement(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _soft_medium()
    level = classify_quality(measure_quality(pixels))
    enhanced, report = pipeline.process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    assert level in {"LOW", "MEDIUM"} or report.quality_level in {"LOW", "MEDIUM"}
    assert report.enhancement_attempted is True
    # Soft photo-like inputs should usually verify classical improvement.
    assert report.enhancement_applied is True
    assert report.enhancement_verified is True
    assert report.enhancement_status in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}
    assert not np.array_equal(pixels, enhanced) or enhanced.shape != pixels.shape
    assert report.after_quality >= report.before_quality - 1e-6


def test_failed_or_invalid_keeps_original(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _high_quality()
    enhanced, report = pipeline.process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    assert report.enhancement_applied is False
    assert np.array_equal(pixels, enhanced)


def test_gate_accepts_blur_reduction_without_quality_regression() -> None:
    from core.contracts.image_quality import ImageQualityMetrics

    before = ImageQualityMetrics(
        resolution_width=400,
        resolution_height=300,
        brightness=0.45,
        contrast=0.40,
        blur_score=0.55,
        noise_score=0.30,
        sharpness=0.35,
        dynamic_range=0.70,
        compression_artifact_score=0.20,
        estimated_quality=0.52,
        motion_blur_score=0.25,
        exposure_score=0.55,
        white_balance_score=0.55,
    )
    after = ImageQualityMetrics(
        resolution_width=400,
        resolution_height=300,
        brightness=0.45,
        contrast=0.41,
        blur_score=0.40,
        noise_score=0.31,
        sharpness=0.36,
        dynamic_range=0.70,
        compression_artifact_score=0.20,
        estimated_quality=0.525,
        motion_blur_score=0.22,
        exposure_score=0.55,
        white_balance_score=0.55,
    )
    gate = compare_metrics(before, after, is_super_resolution=False)
    assert gate.verified is True


def test_disabled_enhancer_keeps_original_and_honest_status() -> None:
    cfg = load_app_config()
    enhancer = ImageEnhancer(cfg)
    pixels = _blurry_low()
    enhanced, report = enhancer.enhance(pixels, enabled=False)
    assert np.array_equal(pixels, enhanced)
    assert report.enhancement_applied is False
    assert report.enhancement_attempted is False
    assert "disabled" in (report.verification_reason or "").lower()
    assert report.enhancement_status != "ENHANCEMENT_APPLIED"
    assert report.enhancement_status != "ENHANCEMENT_NOT_REQUIRED" or report.quality_level == "HIGH"


def test_sr_failure_can_still_apply_classical(tmp_path: Path) -> None:
    """When SR is requested but unavailable, classical enhancement may still verify."""
    cfg = replace(
        DEFAULT_ENHANCEMENT_CONFIG,
        enable_super_resolution=True,
        sr_allow_download=False,
    )
    pipeline = EnhancementPipeline(cfg, models_dir=tmp_path)
    pixels = _soft_medium(h=500, w=700)
    enhanced, report = pipeline.process(
        pixels, competition_mode=False, enable_super_resolution=True
    )
    assert report.quality_level in {"LOW", "MEDIUM"}
    assert report.enhancement_attempted is True
    if report.enhancement_applied:
        assert report.enhancement_verified is True
        assert not np.array_equal(pixels, enhanced) or enhanced.shape != pixels.shape
        assert report.enhancement_status in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}
    else:
        assert np.array_equal(pixels, enhanced)
        assert report.enhancement_status in {
            "ENHANCEMENT_REJECTED",
            "ENHANCEMENT_ATTEMPTED_UNVERIFIED",
            "ENHANCEMENT_FAILED",
        }
