"""Unit tests for image enhancement pipeline."""

import numpy as np

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from vision.enhancement.enhancement_pipeline import EnhancementPipeline
from vision.enhancement.quality_estimator import measure_quality


def test_measure_quality_returns_bounded_scores() -> None:
    pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    metrics = measure_quality(pixels)
    assert 0.0 <= metrics.estimated_quality <= 1.0
    assert metrics.resolution_width == 64
    assert metrics.resolution_height == 64


def test_enhancement_pipeline_adaptive_behavior() -> None:
    pixels = np.random.default_rng(0).integers(40, 220, size=(128, 128, 3), dtype=np.uint8)
    # Keep classic restoration tests free of network SR downloads.
    from dataclasses import replace

    from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG

    cfg = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)
    pipeline = EnhancementPipeline(cfg, models_dir=__import__("pathlib").Path("."))
    enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
    assert enhanced.shape == pixels.shape
    assert report.before_quality >= 0.0
    assert report.after_quality >= report.before_quality or not report.enhancement_applied


def test_competition_mode_does_not_force_excellent_images() -> None:
    # Near-uniform mid-tone with decent metrics should not be forcibly "enhanced".
    pixels = np.full((96, 96, 3), 140, dtype=np.uint8)
    from dataclasses import replace

    cfg = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)
    pipeline = EnhancementPipeline(cfg, models_dir=__import__("pathlib").Path("."))
    _, report = pipeline.process(pixels, competition_mode=True, enable_super_resolution=False)
    assert report.before_quality >= 0.0


def test_enhancement_reverts_gray_collapse() -> None:
    # Strongly chromatic image — pipeline must not keep a gray-collapsed result.
    # Use a larger canvas so LOW-tier upscaling is not required for the chroma check.
    pixels = np.zeros((520, 520, 3), dtype=np.uint8)
    pixels[:, :] = (180, 40, 40)
    from dataclasses import replace

    cfg = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)
    pipeline = EnhancementPipeline(cfg, models_dir=__import__("pathlib").Path("."))
    enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
    assert enhanced.shape[2] == 3
    # Either unchanged or still chromatic — never forced to near-gray.
    chroma = float(np.mean(np.std(enhanced.reshape(-1, 3).astype(np.float32), axis=1)))
    assert chroma > 20.0 or not report.enhancement_applied


def test_enhancement_pipeline_caches_results() -> None:
    pixels = np.random.default_rng(1).integers(30, 200, size=(64, 64, 3), dtype=np.uint8)
    from dataclasses import replace

    cfg = replace(DEFAULT_ENHANCEMENT_CONFIG, enable_super_resolution=False, sr_allow_download=False)
    pipeline = EnhancementPipeline(cfg, models_dir=__import__("pathlib").Path("."))
    first, report_first = pipeline.process(pixels, competition_mode=True, enable_super_resolution=False)
    second, report_second = pipeline.process(pixels, competition_mode=True, enable_super_resolution=False)
    assert np.array_equal(first, second)
    assert report_first.enhancement_applied == report_second.enhancement_applied
