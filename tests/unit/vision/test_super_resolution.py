"""Tests for real super-resolution path and enhancement gating."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from core.config.enhancement_config import EnhancementConfig
from vision.enhancement.enhancement_pipeline import EnhancementPipeline
from vision.enhancement.quality_estimator import classify_quality, measure_quality
from vision.enhancement.super_resolution import (
    UpscaleResult,
    lanczos_upscale,
    validate_sr_against_baseline,
)


def _cfg(**overrides: object) -> EnhancementConfig:
    base = DEFAULT_ENHANCEMENT_CONFIG.__dict__.copy()
    base.update(overrides)
    return EnhancementConfig(**base)  # type: ignore[arg-type]


def test_high_quality_skips_enhancement() -> None:
    # Large, sharp synthetic grid → HIGH → original kept.
    yy, xx = np.mgrid[0:640, 0:640]
    pixels = ((xx + yy) % 32 < 16).astype(np.uint8) * 200
    pixels = np.stack([pixels, pixels, pixels], axis=-1)
    metrics = measure_quality(pixels)
    # Force HIGH path via pipeline: if not HIGH, still must not claim SR without mock.
    pipeline = EnhancementPipeline(_cfg(enable_super_resolution=True), models_dir=Path("."))
    with patch("vision.enhancement.enhancement_pipeline.classify_quality", return_value="HIGH"):
        enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
    assert report.quality_level == "HIGH"
    assert report.enhancement_applied is False
    assert report.super_resolution_used is False
    assert enhanced.shape == pixels.shape


def test_low_quality_attempts_sr() -> None:
    pixels = np.random.default_rng(0).integers(40, 180, size=(200, 300, 3), dtype=np.uint8)
    fake = UpscaleResult(
        pixels=lanczos_upscale(pixels, 2),
        backend="realesrgan",
        model_name="RealESRGAN_x2plus",
        scale=2,
        device="cpu",
        tile_size=128,
        input_size=(300, 200),
        output_size=(600, 400),
        true_sr=True,
        message="ok",
    )
    pipeline = EnhancementPipeline(_cfg(enable_super_resolution=True), models_dir=Path("."))
    with (
        patch("vision.enhancement.enhancement_pipeline.classify_quality", return_value="LOW"),
        patch("vision.enhancement.enhancement_pipeline.upscale", return_value=fake),
        patch(
            "vision.enhancement.enhancement_pipeline.validate_sr_against_baseline",
            return_value=(True, "detail_gain_vs_lanczos=0.20"),
        ),
    ):
        enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
    assert report.enhancement_attempted is True
    assert report.super_resolution_used is True
    assert report.enhancement_applied is True
    assert enhanced.shape[0] == 400 and enhanced.shape[1] == 600
    assert report.sr_model == "RealESRGAN_x2plus"
    assert report.sr_scale == 2


def test_medium_quality_attempts_sr() -> None:
    pixels = np.random.default_rng(1).integers(50, 200, size=(360, 480, 3), dtype=np.uint8)
    fake = UpscaleResult(
        pixels=lanczos_upscale(pixels, 2),
        backend="realesrgan",
        model_name="realesr-general-x4v3",
        scale=2,
        device="cpu",
        tile_size=128,
        input_size=(480, 360),
        output_size=(960, 720),
        true_sr=True,
        message="ok",
    )
    pipeline = EnhancementPipeline(_cfg(enable_super_resolution=True), models_dir=Path("."))
    with (
        patch("vision.enhancement.enhancement_pipeline.classify_quality", return_value="MEDIUM"),
        patch("vision.enhancement.enhancement_pipeline.upscale", return_value=fake),
        patch(
            "vision.enhancement.enhancement_pipeline.validate_sr_against_baseline",
            return_value=(True, "detail_gain_vs_lanczos=0.15"),
        ),
    ):
        enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
    assert report.super_resolution_used is True
    assert report.enhancement_applied is True
    assert enhanced.shape != pixels.shape


def test_sr_validation_failure_keeps_original() -> None:
    pixels = np.random.default_rng(2).integers(30, 200, size=(180, 240, 3), dtype=np.uint8)
    fake = UpscaleResult(
        pixels=lanczos_upscale(pixels, 2),
        backend="realesrgan",
        model_name="RealESRGAN_x2plus",
        scale=2,
        device="cpu",
        tile_size=128,
        input_size=(240, 180),
        output_size=(480, 360),
        true_sr=True,
        message="ok",
    )
    pipeline = EnhancementPipeline(_cfg(enable_super_resolution=True), models_dir=Path("."))
    with (
        patch("vision.enhancement.enhancement_pipeline.classify_quality", return_value="LOW"),
        patch("vision.enhancement.enhancement_pipeline.upscale", return_value=fake),
        patch(
            "vision.enhancement.enhancement_pipeline.validate_sr_against_baseline",
            return_value=(False, "no measurable detail gain vs Lanczos"),
        ),
    ):
        enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
    assert report.super_resolution_used is False
    assert enhanced.shape == pixels.shape or not report.enhancement_applied or "clarity" in " ".join(
        report.enhancement_operations
    )


def test_sr_cache_reuses_result() -> None:
    pixels = np.random.default_rng(3).integers(40, 190, size=(160, 200, 3), dtype=np.uint8)
    fake = UpscaleResult(
        pixels=lanczos_upscale(pixels, 2),
        backend="realesrgan",
        model_name="RealESRGAN_x2plus",
        scale=2,
        device="cpu",
        tile_size=128,
        input_size=(200, 160),
        output_size=(400, 320),
        true_sr=True,
        message="ok",
    )
    pipeline = EnhancementPipeline(_cfg(enable_super_resolution=True), models_dir=Path("."))
    with (
        patch("vision.enhancement.enhancement_pipeline.classify_quality", return_value="LOW"),
        patch(
            "vision.enhancement.enhancement_pipeline.upscale",
            return_value=fake,
        ) as mocked,
        patch(
            "vision.enhancement.enhancement_pipeline.validate_sr_against_baseline",
            return_value=(True, "ok"),
        ),
    ):
        first, r1 = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
        second, r2 = pipeline.process(pixels, competition_mode=False, enable_super_resolution=True)
    assert mocked.call_count == 1
    assert np.array_equal(first, second)
    assert r1.super_resolution_used == r2.super_resolution_used is True


def test_validate_sr_rejects_same_size() -> None:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, reason = validate_sr_against_baseline(pixels, pixels, scale=2)
    assert ok is False
    assert "resolution" in reason


def test_cuda_oom_falls_back_to_cpu(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """When CUDA inference fails, CPU retry path is attempted."""
    from vision.enhancement import super_resolution as sr

    calls: list[str] = []

    def boom_then_cpu(**kwargs):  # type: ignore[no-untyped-def]
        device = kwargs.get("device")
        calls.append(str(device))
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        pixels = kwargs["pixels"] if "pixels" in kwargs else None
        # _infer signature uses positional pixels — handled below via patch target.
        raise AssertionError("should patch _infer_realesrgan")

    pixels = np.random.default_rng(4).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    weights = Path("dummy.pth")

    def fake_infer(pixels, *, weights, kind, outscale, tile_size, tile_overlap, device):  # noqa: ANN001
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return lanczos_upscale(pixels, outscale)

    with (
        patch.object(sr, "_ensure_weights", return_value=weights),
        patch.object(sr, "_resolve_device", return_value="cuda"),
        patch.object(sr, "_infer_realesrgan", side_effect=fake_infer),
    ):
        result = sr._try_realesrgan(
            pixels,
            models_dir=Path("."),
            scale=2,
            tile_size=64,
            tile_overlap=4,
            device_pref="cuda",
            allow_download=False,
        )
    assert result is not None
    assert result.device == "cpu"
    assert "cuda" in calls and "cpu" in calls
