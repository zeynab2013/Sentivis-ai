"""PART 2-A — Enhancement honesty / multi-signal verification regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
from core.contracts.image_quality import ImageQualityMetrics
from vision.enhancement.enhancement_gate import compare_metrics
from vision.enhancement.enhancement_pipeline import EnhancementPipeline
from vision.enhancement.quality_estimator import classify_quality, measure_quality

_NO_SR = replace(
    DEFAULT_ENHANCEMENT_CONFIG,
    enable_super_resolution=False,
    sr_allow_download=False,
)


def _metrics(**overrides: float | int) -> ImageQualityMetrics:
    base = dict(
        resolution_width=320,
        resolution_height=240,
        brightness=0.45,
        contrast=0.40,
        blur_score=0.35,
        noise_score=0.30,
        sharpness=0.40,
        dynamic_range=0.70,
        compression_artifact_score=0.20,
        estimated_quality=0.55,
        motion_blur_score=0.20,
        exposure_score=0.55,
        white_balance_score=0.55,
    )
    base.update(overrides)
    return ImageQualityMetrics(**base)  # type: ignore[arg-type]


def _blurry_low(h: int = 180, w: int = 240) -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.integers(20, 90, size=(h, w, 3), dtype=np.uint8)
    soft = base.copy()
    for _ in range(4):
        soft[1:, 1:] = (
            (soft[1:, 1:].astype(np.uint16) + soft[:-1, :-1].astype(np.uint16)) // 2
        ).astype(np.uint8)
    return soft


def _compressed_like(h: int = 200, w: int = 280) -> np.ndarray:
    """Blocky / soft image approximating JPEG damage."""
    rng = np.random.default_rng(21)
    img = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    # 8x8 block average to mimic compression.
    out = img.copy()
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = img[y : y + 8, x : x + 8].mean(axis=(0, 1)).astype(np.uint8)
            out[y : y + 8, x : x + 8] = block
    return out


def _medium_quality(h: int = 400, w: int = 560) -> np.ndarray:
    rng = np.random.default_rng(31)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (120, 130, 140)
    # Mild structure.
    for i in range(8):
        y0 = 40 + i * 40
        x0 = 40 + i * 50
        img[y0 : y0 + 30, x0 : x0 + 80] = rng.integers(60, 200, size=3)
    return img


def _high_quality(h: int = 640, w: int = 800) -> np.ndarray:
    """Crisp high-contrast grid that classifies as HIGH."""
    yy, xx = np.mgrid[0:h, 0:w]
    pixels = ((xx + yy) % 32 < 16).astype(np.uint8) * 200
    return np.stack([pixels, pixels, pixels], axis=-1)


def test_gate_rejects_quality_regression() -> None:
    before = _metrics(estimated_quality=0.70, sharpness=0.50, blur_score=0.30, noise_score=0.25)
    after = _metrics(estimated_quality=0.55, sharpness=0.35, blur_score=0.45, noise_score=0.40)
    gate = compare_metrics(before, after, is_super_resolution=False)
    assert gate.verified is False
    assert "reject" in gate.reason.lower() or "regress" in gate.reason.lower()


def test_gate_rejects_oversharpen_noise() -> None:
    before = _metrics(estimated_quality=0.60, sharpness=0.40, noise_score=0.25)
    after = _metrics(estimated_quality=0.61, sharpness=0.55, noise_score=0.40)
    gate = compare_metrics(before, after, is_super_resolution=False)
    assert gate.verified is False


def test_gate_accepts_multi_signal_gain() -> None:
    before = _metrics(estimated_quality=0.50, sharpness=0.30, blur_score=0.50, noise_score=0.35)
    after = _metrics(estimated_quality=0.58, sharpness=0.42, blur_score=0.38, noise_score=0.30)
    gate = compare_metrics(before, after, is_super_resolution=False)
    assert gate.verified is True


def test_gate_sr_requires_resolution_increase() -> None:
    before = _metrics(resolution_width=200, resolution_height=150, estimated_quality=0.50)
    after = _metrics(resolution_width=200, resolution_height=150, estimated_quality=0.60)
    gate = compare_metrics(before, after, baseline=before, is_super_resolution=True)
    assert gate.verified is False
    assert "dimension" in gate.reason.lower() or "resolution" in gate.reason.lower()


def test_gate_sr_accepts_non_regressing_upscale() -> None:
    before = _metrics(resolution_width=100, resolution_height=80, estimated_quality=0.50, sharpness=0.35)
    baseline = _metrics(resolution_width=200, resolution_height=160, estimated_quality=0.52, sharpness=0.36)
    after = _metrics(resolution_width=200, resolution_height=160, estimated_quality=0.53, sharpness=0.37)
    gate = compare_metrics(before, after, baseline=baseline, is_super_resolution=True)
    assert gate.verified is True
    assert gate.resolution_increased is True


def test_gate_sr_rejects_worse_than_lanczos() -> None:
    before = _metrics(resolution_width=100, resolution_height=80, estimated_quality=0.50, sharpness=0.40)
    baseline = _metrics(resolution_width=200, resolution_height=160, estimated_quality=0.55, sharpness=0.42)
    after = _metrics(resolution_width=200, resolution_height=160, estimated_quality=0.45, sharpness=0.30)
    gate = compare_metrics(before, after, baseline=baseline, is_super_resolution=True)
    assert gate.verified is False


def test_high_quality_not_needed(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _high_quality()
    enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
    assert report.quality_level == "HIGH"
    assert report.enhancement_attempted is False
    assert report.enhancement_applied is False
    assert report.enhancement_verified is False
    assert report.enhancement_status == "ENHANCEMENT_NOT_REQUIRED"
    assert "sufficient" in (report.verification_reason or "").lower()
    assert np.array_equal(pixels, enhanced)


def test_low_quality_is_attempted(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    pixels = _blurry_low()
    _enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
    assert report.quality_level in {"LOW", "MEDIUM"}
    assert report.enhancement_attempted is True
    # Applied only if verified — never claim success without verification.
    if report.enhancement_applied:
        assert report.enhancement_verified is True
        assert report.enhancement_status in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}
        assert report.after_quality + 1e-6 >= report.before_quality - 0.03
    else:
        assert report.enhancement_verified is False
        assert report.enhancement_status in {
            "ENHANCEMENT_REJECTED",
            "ENHANCEMENT_ATTEMPTED_UNVERIFIED",
            "ENHANCEMENT_FAILED",
        }
        assert report.verification_reason or report.rejection_reason


def test_worse_candidate_never_accepted(tmp_path: Path) -> None:
    """Force a degrading candidate via gate unit; pipeline must keep original."""
    before = measure_quality(_medium_quality())
    worse = _metrics(
        resolution_width=before.resolution_width,
        resolution_height=before.resolution_height,
        estimated_quality=max(0.05, before.estimated_quality - 0.20),
        sharpness=max(0.05, before.sharpness - 0.20),
        blur_score=min(0.95, before.blur_score + 0.20),
        noise_score=min(0.95, before.noise_score + 0.15),
        brightness=before.brightness,
        contrast=before.contrast,
        dynamic_range=before.dynamic_range,
        compression_artifact_score=before.compression_artifact_score,
    )
    gate = compare_metrics(before, worse, is_super_resolution=False)
    assert gate.verified is False


def test_benchmark_cases_record_metadata(tmp_path: Path) -> None:
    """Deterministic LOW/MEDIUM/HIGH/COMPRESSED cases — no fabricated % expectations."""
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    cases = {
        "low_blurry": _blurry_low(),
        "compressed": _compressed_like(),
        "medium": _medium_quality(),
        "high": _high_quality(),
        "low_tiny": _blurry_low(120, 160),
    }
    rows: list[dict[str, object]] = []
    for name, pixels in cases.items():
        enhanced, report = pipeline.process(
            pixels, competition_mode=False, enable_super_resolution=False
        )
        level = classify_quality(measure_quality(pixels))
        rows.append(
            {
                "name": name,
                "level": report.quality_level,
                "classified": level,
                "orig_wh": (report.original_width, report.original_height),
                "out_wh": (report.output_width, report.output_height),
                "attempted": report.enhancement_attempted,
                "verified": report.enhancement_verified,
                "applied": report.enhancement_applied,
                "status": report.enhancement_status,
                "reason": report.verification_reason or report.rejection_reason,
                "before_q": report.before_quality,
                "after_q": report.after_quality,
                "delta": report.quality_delta_percent,
                "improvement": report.improvement_percent,
                "pixels_changed": not np.array_equal(pixels, enhanced),
            }
        )
        # Honesty invariants
        if report.enhancement_applied:
            assert report.enhancement_verified
            assert report.improvement_percent >= 0.0
            if not report.super_resolution_used:
                assert not np.array_equal(pixels, enhanced)
        else:
            assert np.array_equal(pixels, enhanced)
            assert report.improvement_percent == 0.0
        if report.quality_level == "HIGH":
            assert report.enhancement_attempted is False
            assert report.enhancement_status == "ENHANCEMENT_NOT_REQUIRED"

    attempted = sum(1 for r in rows if r["attempted"])
    verified = sum(1 for r in rows if r["verified"])
    rejected = sum(
        1
        for r in rows
        if r["attempted"] and not r["verified"]
    )
    assert attempted >= 1
    assert verified + rejected == attempted
    # High-quality case must be present and skipped.
    assert any(r["name"] == "high" and not r["attempted"] for r in rows)


def test_applied_implies_verified_and_not_worse(tmp_path: Path) -> None:
    pipeline = EnhancementPipeline(_NO_SR, models_dir=tmp_path)
    for pixels in (_blurry_low(), _compressed_like(), _medium_quality()):
        enhanced, report = pipeline.process(
            pixels, competition_mode=False, enable_super_resolution=False
        )
        if report.enhancement_applied:
            assert report.enhancement_verified is True
            assert report.enhancement_status in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}
            assert report.after_quality + 0.03 >= report.before_quality
            assert not np.array_equal(pixels, enhanced)
        else:
            assert np.array_equal(pixels, enhanced)
