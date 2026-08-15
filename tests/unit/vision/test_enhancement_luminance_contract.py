"""Enhancement must not unexpectedly darken well-exposed images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from vision.enhancement.enhancement_pipeline import EnhancementPipeline
from vision.enhancement.luminance import mean_luminance
from vision.enhancement.quality_estimator import classify_quality, measure_quality


def _pipeline() -> EnhancementPipeline:
    from dataclasses import replace

    cfg = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)
    return EnhancementPipeline(cfg, models_dir=Path("."))


def test_bright_low_image_is_not_force_darkened() -> None:
    # Soft / low-detail but bright — historically crushed toward mid-gray.
    rng = np.random.default_rng(11)
    pixels = rng.integers(150, 220, size=(160, 160, 3), dtype=np.uint8)
    pixels = ((pixels.astype(np.float32) * 0.7 + np.roll(pixels, 1, axis=0) * 0.3)).astype(np.uint8)
    before = mean_luminance(pixels)
    assert before > 0.45
    enhanced, report = _pipeline().process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    after = mean_luminance(enhanced)
    assert after + 0.04 >= before
    if report.enhancement_applied:
        assert after + 0.03 >= before


def test_normal_image_preserves_exposure() -> None:
    rng = np.random.default_rng(3)
    pixels = rng.integers(70, 200, size=(256, 256, 3), dtype=np.uint8)
    pixels[20:40, 20:40] = 255
    pixels[60:70, 60:200] = 10
    before = mean_luminance(pixels)
    enhanced, report = _pipeline().process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    after = mean_luminance(enhanced)
    assert abs(after - before) < 0.08 or not report.enhancement_applied


def test_dark_image_may_brighten_but_not_collapse() -> None:
    rng = np.random.default_rng(5)
    pixels = rng.integers(5, 40, size=(200, 200, 3), dtype=np.uint8)
    before = mean_luminance(pixels)
    enhanced, _report = _pipeline().process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    after = mean_luminance(enhanced)
    assert after + 0.02 >= before


def test_noisy_blurry_low_does_not_accept_darkened_result() -> None:
    rng = np.random.default_rng(9)
    base = rng.integers(90, 170, size=(128, 128, 3), dtype=np.uint8)
    soft = base.copy()
    soft[1:, 1:] = ((soft[1:, 1:].astype(np.uint16) + soft[:-1, :-1].astype(np.uint16)) // 2).astype(
        np.uint8
    )
    level = classify_quality(measure_quality(soft))
    enhanced, report = _pipeline().process(
        soft, competition_mode=False, enable_super_resolution=False
    )
    before = mean_luminance(soft)
    after = mean_luminance(enhanced)
    assert after + 0.04 >= before
    if level == "HIGH":
        assert report.enhancement_applied is False


def test_rgb_channel_order_preserved_on_red_patch() -> None:
    pixels = np.zeros((96, 96, 3), dtype=np.uint8)
    pixels[:, :] = (200, 40, 40)
    enhanced, report = _pipeline().process(
        pixels, competition_mode=False, enable_super_resolution=False
    )
    # Red channel must remain dominant (no accidental BGR swap / gray cast).
    mean_rgb = enhanced.reshape(-1, 3).mean(axis=0)
    assert float(mean_rgb[0]) > float(mean_rgb[1]) + 40
    assert float(mean_rgb[0]) > float(mean_rgb[2]) + 40
    chroma = float(np.mean(np.std(enhanced.reshape(-1, 3).astype(np.float32), axis=1)))
    assert chroma > 20.0 or not report.enhancement_applied
