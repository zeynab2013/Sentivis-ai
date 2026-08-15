"""Image quality assessment and enhancement report DTOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Measured quality dimensions before enhancement."""

    resolution_width: int
    resolution_height: int
    brightness: float
    contrast: float
    blur_score: float
    noise_score: float
    sharpness: float
    dynamic_range: float
    compression_artifact_score: float
    estimated_quality: float
    motion_blur_score: float = 0.0
    exposure_score: float = 0.5
    white_balance_score: float = 0.5


@dataclass(frozen=True)
class ImageQualityReport:
    """Complete image quality assessment and enhancement outcome.

    Three-state honesty:
      ENHANCEMENT_NOT_REQUIRED — input already sufficient (not attempted)
      ENHANCEMENT_ATTEMPTED_UNVERIFIED — ran model/ops but not verified better
      ENHANCEMENT_VERIFIED / ENHANCEMENT_APPLIED — verified improvement accepted

    ``enhancement_applied`` is True ONLY when the candidate is verified.
    """

    metrics: ImageQualityMetrics
    enhancement_operations: tuple[str, ...]
    enhancement_applied: bool
    processing_time_ms: float
    improvement_percent: float
    before_quality: float
    after_quality: float
    super_resolution_used: bool = False
    sam2_available: bool = False
    # HIGH | MEDIUM | LOW — classified from original image before enhancement.
    quality_level: str = "MEDIUM"
    # Explicit attempt / reject diagnostics for UI honesty.
    enhancement_attempted: bool = False
    enhancement_rejected: bool = False
    rejection_reason: str = ""
    # Exact status: NOT_REQUIRED | APPLIED/VERIFIED | FAILED | REJECTED | ATTEMPTED_UNVERIFIED
    enhancement_status: str = "ENHANCEMENT_NOT_REQUIRED"
    # Honest SR diagnostics (empty when SR was not used).
    sr_model: str = ""
    sr_scale: int = 1
    sr_device: str = ""
    sr_input_size: str = ""
    sr_output_size: str = ""
    # PART 2-A: verification metadata
    enhancement_verified: bool = False
    verification_reason: str = ""
    quality_delta_percent: float = 0.0  # signed; may be negative when rejected
    original_width: int = 0
    original_height: int = 0
    output_width: int = 0
    output_height: int = 0
    before_sharpness: float = 0.0
    after_sharpness: float = 0.0
    before_blur_score: float = 0.0
    after_blur_score: float = 0.0
    before_noise_score: float = 0.0
    after_noise_score: float = 0.0
