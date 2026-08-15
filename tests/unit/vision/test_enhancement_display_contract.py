"""Enhancement must change pixels when accepted and report reject honestly."""

from __future__ import annotations

import hashlib

import numpy as np

from dataclasses import replace

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from vision.enhancement.enhancement_pipeline import EnhancementPipeline

_NO_SR = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)


def _hash(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def _dark_blurry(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(8, 36, size=(h, w, 3), dtype=np.uint8)
    # Soften heavily to encourage enhancement.
    soft = base.copy()
    soft[1:, 1:] = ((soft[1:, 1:].astype(np.uint16) + soft[:-1, :-1].astype(np.uint16)) // 2).astype(
        np.uint8
    )
    return soft


def test_low_quality_enhancement_changes_pixels_when_accepted(tmp_path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    original = _dark_blurry()
    enhanced, report = pipeline.process(original, competition_mode=False, enable_super_resolution=False)
    assert report.quality_level in {"LOW", "MEDIUM", "HIGH"}
    assert report.enhancement_attempted is (report.quality_level != "HIGH")
    if report.enhancement_applied:
        assert _hash(original) != _hash(enhanced)
        assert report.enhancement_rejected is False
        assert report.after_quality + 1e-6 >= report.before_quality
    else:
        # Honest reject path: original preserved.
        assert np.array_equal(original, enhanced)
        if report.enhancement_attempted:
            assert report.enhancement_rejected is True
            assert report.rejection_reason


def test_high_quality_keeps_original(tmp_path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    # Bright, sharp synthetic image.
    pixels = np.full((480, 640, 3), 180, dtype=np.uint8)
    pixels[40:80, 40:80] = 20
    enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
    if report.quality_level == "HIGH":
        assert report.enhancement_applied is False
        assert report.enhancement_attempted is False
        assert np.array_equal(pixels, enhanced)
