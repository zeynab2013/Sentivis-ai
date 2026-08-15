"""Image quality measurement utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.contracts.image_quality import ImageQualityMetrics

QualityLevel = Literal["HIGH", "MEDIUM", "LOW"]


def measure_quality(pixels: NDArray[np.uint8]) -> ImageQualityMetrics:
    """Estimate image quality metrics from RGB uint8 array."""
    height, width = pixels.shape[:2]
    gray = _to_grayscale(pixels)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 128.0)
    blur_score = _blur_score(gray)
    motion_blur_score = _motion_blur_score(gray)
    noise_score = _noise_score(gray)
    sharpness = 1.0 - max(blur_score, motion_blur_score * 0.85)
    dynamic_range = float((float(np.max(gray)) - float(np.min(gray))) / 255.0)
    compression_artifact_score = _compression_artifact_score(gray)
    exposure_score = _exposure_score(brightness, dynamic_range)
    white_balance_score = _white_balance_score(pixels)
    estimated = _aggregate_quality(
        brightness,
        contrast,
        blur_score,
        motion_blur_score,
        noise_score,
        sharpness,
        dynamic_range,
        compression_artifact_score,
        exposure_score,
        white_balance_score,
    )
    return ImageQualityMetrics(
        resolution_width=width,
        resolution_height=height,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        noise_score=noise_score,
        sharpness=sharpness,
        dynamic_range=dynamic_range,
        compression_artifact_score=compression_artifact_score,
        estimated_quality=estimated,
        motion_blur_score=motion_blur_score,
        exposure_score=exposure_score,
        white_balance_score=white_balance_score,
    )


def needs_enhancement(metrics: ImageQualityMetrics, *, threshold: float) -> bool:
    """Return True when estimated quality is below threshold."""
    return metrics.estimated_quality < threshold


def classify_quality(metrics: ImageQualityMetrics) -> QualityLevel:
    """Classify visual quality into HIGH / MEDIUM / LOW for enhancement gating.

    Uses resolution, blur, sharpness, noise, brightness/contrast proxies, and the
    aggregate estimated_quality score. HIGH images must not be enhanced.
    """
    min_dim = min(metrics.resolution_width, metrics.resolution_height)
    megapixels = (metrics.resolution_width * metrics.resolution_height) / 1_000_000.0
    q = metrics.estimated_quality

    severe_blur = metrics.blur_score >= 0.70 or metrics.motion_blur_score >= 0.55
    severe_noise = metrics.noise_score >= 0.62
    poor_exposure = metrics.brightness < 0.18 or metrics.brightness > 0.88
    tiny = min_dim < 220 or megapixels < 0.05
    soft = metrics.sharpness < 0.16

    if tiny or severe_blur or (severe_noise and soft) or (q < 0.42) or (poor_exposure and q < 0.55):
        return "LOW"

    high_ready = (
        q >= 0.78
        and min_dim >= 480
        and metrics.blur_score < 0.42
        and metrics.sharpness >= 0.38
        and metrics.noise_score < 0.48
        and 0.22 <= metrics.brightness <= 0.82
        and metrics.contrast >= 0.22
    )
    if high_ready:
        return "HIGH"

    if q < 0.55 or min_dim < 360 or metrics.blur_score >= 0.58 or metrics.sharpness < 0.24:
        return "LOW"

    return "MEDIUM"


def _to_grayscale(pixels: NDArray[np.uint8]) -> NDArray[np.float32]:
    rgb = pixels.astype(np.float32)
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.asarray(gray, dtype=np.float32)


def _blur_score(gray: NDArray[np.float32]) -> float:
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
    )
    variance = float(np.var(laplacian))
    return float(max(0.0, min(1.0, 1.0 - variance / 500.0)))


def _motion_blur_score(gray: NDArray[np.float32]) -> float:
    """Directional gradient anisotropy as a proxy for motion blur."""
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return 0.0
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    denom = max(gx + gy, 1e-6)
    anisotropy = abs(gx - gy) / denom
    # High anisotropy + low total edge energy suggests streaking.
    energy = float((gx + gy) / 64.0)
    score = anisotropy * (1.0 - min(1.0, energy))
    return float(max(0.0, min(1.0, score)))


def _noise_score(gray: NDArray[np.float32]) -> float:
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return 0.0
    diff = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    return float(max(0.0, min(1.0, float(np.mean(diff)) / 64.0)))


def _compression_artifact_score(gray: NDArray[np.float32]) -> float:
    if gray.shape[1] < 16:
        return 0.0
    block = gray[:, : gray.shape[1] // 8 * 8].reshape(gray.shape[0], -1, 8)
    block_var = np.var(block, axis=2)
    return float(max(0.0, min(1.0, 1.0 - float(np.mean(block_var)) / 400.0)))


def _exposure_score(brightness: float, dynamic_range: float) -> float:
    # 1.0 = well exposed; lower when crushed or blown.
    centered = 1.0 - abs(brightness - 0.5) * 1.6
    return float(max(0.0, min(1.0, 0.65 * centered + 0.35 * dynamic_range)))


def _white_balance_score(pixels: NDArray[np.uint8]) -> float:
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return 0.5
    means = pixels.reshape(-1, pixels.shape[2])[:, :3].astype(np.float32).mean(axis=0)
    avg = float(means.mean()) + 1e-6
    channel_dev = float(np.mean(np.abs(means - avg)) / avg)
    return float(max(0.0, min(1.0, 1.0 - channel_dev * 2.5)))


def _aggregate_quality(
    brightness: float,
    contrast: float,
    blur_score: float,
    motion_blur_score: float,
    noise_score: float,
    sharpness: float,
    dynamic_range: float,
    compression_artifact_score: float,
    exposure_score: float,
    white_balance_score: float,
) -> float:
    brightness_penalty = abs(brightness - 0.5) * 0.35
    score = (
        0.18 * contrast
        + 0.18 * sharpness
        + 0.12 * dynamic_range
        + 0.12 * (1.0 - blur_score)
        + 0.08 * (1.0 - motion_blur_score)
        + 0.08 * (1.0 - noise_score)
        + 0.08 * (1.0 - compression_artifact_score)
        + 0.08 * exposure_score
        + 0.08 * white_balance_score
        - brightness_penalty
    )
    return float(max(0.0, min(1.0, score)))
