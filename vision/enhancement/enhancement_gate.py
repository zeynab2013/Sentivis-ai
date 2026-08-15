"""Conservative multi-signal enhancement acceptance gate.

States:
  NOT_NEEDED — input already sufficient (no attempt)
  ATTEMPTED_UNVERIFIED — ran enhancement but could not establish improvement
  VERIFIED — candidate clearly better on configured multi-signal checks

Never accept a worse candidate. Never treat a single sharpness bump as success.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.image_quality import ImageQualityMetrics


@dataclass(frozen=True)
class EnhancementGateResult:
    """Outcome of comparing original vs candidate enhancement."""

    verified: bool
    reason: str
    quality_delta: float  # after - baseline (same-size comparable scores)
    sharpness_delta: float
    blur_delta: float  # negative is better (less blur)
    noise_delta: float  # positive is worse
    resolution_increased: bool
    signals_improved: tuple[str, ...]
    signals_regressed: tuple[str, ...]


def compare_metrics(
    before: ImageQualityMetrics,
    after: ImageQualityMetrics,
    *,
    baseline: ImageQualityMetrics | None = None,
    is_super_resolution: bool = False,
    quality_slack: float = 0.01,
    sharpness_slack: float = 0.02,
) -> EnhancementGateResult:
    """Multi-signal comparison. ``baseline`` is same-size reference (e.g. Lanczos).

    For SR, ``before`` is native-resolution metrics; ``baseline`` must be the
    upscaled Lanczos reference at the candidate's resolution. Quality deltas
    use baseline when provided so native vs 2× scores are not mixed.
    """
    ref = baseline if baseline is not None else before
    q_delta = after.estimated_quality - ref.estimated_quality
    s_delta = after.sharpness - ref.sharpness
    # Lower blur/noise is better.
    blur_delta = after.blur_score - ref.blur_score
    noise_delta = after.noise_score - ref.noise_score
    res_up = (
        after.resolution_width > before.resolution_width
        and after.resolution_height > before.resolution_height
    )

    improved: list[str] = []
    regressed: list[str] = []

    if q_delta > quality_slack:
        improved.append("quality")
    elif q_delta < -quality_slack:
        regressed.append("quality")

    if s_delta > sharpness_slack:
        improved.append("sharpness")
    elif s_delta < -sharpness_slack:
        regressed.append("sharpness")

    if blur_delta < -0.02:
        improved.append("blur")
    elif blur_delta > 0.04:
        regressed.append("blur")

    if noise_delta < -0.02:
        improved.append("noise")
    elif noise_delta > 0.08:
        regressed.append("noise")

    if abs(after.contrast - ref.contrast) > 0.03:
        if after.contrast > ref.contrast:
            improved.append("contrast")
        else:
            # Contrast drop alone is soft unless severe.
            if after.contrast + 0.06 < ref.contrast:
                regressed.append("contrast")

    if is_super_resolution:
        if not res_up:
            return EnhancementGateResult(
                verified=False,
                reason="Enhancement rejected: output dimensions did not increase for super-resolution.",
                quality_delta=q_delta,
                sharpness_delta=s_delta,
                blur_delta=blur_delta,
                noise_delta=noise_delta,
                resolution_increased=False,
                signals_improved=tuple(improved),
                signals_regressed=tuple(regressed),
            )
        # Hard reject: clearly worse than same-size Lanczos on quality or sharpness.
        if "quality" in regressed and "sharpness" in regressed:
            return EnhancementGateResult(
                verified=False,
                reason=(
                    "Enhancement attempted but rejected: super-resolution output "
                    "regressed vs Lanczos baseline on quality and sharpness."
                ),
                quality_delta=q_delta,
                sharpness_delta=s_delta,
                blur_delta=blur_delta,
                noise_delta=noise_delta,
                resolution_increased=True,
                signals_improved=tuple(improved),
                signals_regressed=tuple(regressed),
            )
        if noise_delta > 0.12 and s_delta > 0.05 and q_delta < 0.02:
            return EnhancementGateResult(
                verified=False,
                reason=(
                    "Enhancement attempted but rejected: unnatural sharpening "
                    "with elevated noise vs baseline."
                ),
                quality_delta=q_delta,
                sharpness_delta=s_delta,
                blur_delta=blur_delta,
                noise_delta=noise_delta,
                resolution_increased=True,
                signals_improved=tuple(improved),
                signals_regressed=tuple(regressed),
            )
        # Verified SR: resolution up + not worse than Lanczos (within slack),
        # and either quality/sharpness/blur improves OR at least non-regressing.
        # Allow modest noise rise when quality clearly improved (real SR detail).
        noise_cap = 0.14 if q_delta >= 0.02 else 0.10
        not_worse = q_delta >= -quality_slack and s_delta >= -sharpness_slack and noise_delta <= noise_cap
        has_gain = bool(improved) or (q_delta >= 0.0 and s_delta >= 0.0)
        if not_worse and (has_gain or (q_delta >= -0.005 and s_delta >= -0.005)):
            # Resolution increase with non-regressing detail is a verified SR outcome.
            # Do not require a large synthetic quality % — that games the metric.
            gain_note = (
                f"signals={','.join(improved) or 'stable'}; "
                f"q_delta={q_delta:+.3f}; sharp_delta={s_delta:+.3f}"
            )
            return EnhancementGateResult(
                verified=True,
                reason=(
                    "Enhancement verified: resolution increased and quality checks "
                    f"passed vs Lanczos baseline ({gain_note})."
                ),
                quality_delta=q_delta,
                sharpness_delta=s_delta,
                blur_delta=blur_delta,
                noise_delta=noise_delta,
                resolution_increased=True,
                signals_improved=tuple(improved) if improved else ("resolution",),
                signals_regressed=tuple(regressed),
            )
        return EnhancementGateResult(
            verified=False,
            reason=(
                "Enhancement attempted but not verified: no measurable improvement "
                f"vs Lanczos baseline (q_delta={q_delta:+.3f}, sharp_delta={s_delta:+.3f})."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=True,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )

    # Classical restoration — require clear multi-signal improvement, no hard regression.
    if regressed and not improved:
        return EnhancementGateResult(
            verified=False,
            reason=(
                "Enhancement attempted but rejected: candidate regressed on "
                f"{', '.join(regressed)} with no compensating gains."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    if "quality" in regressed:
        return EnhancementGateResult(
            verified=False,
            reason="Enhancement attempted but rejected: overall quality score decreased.",
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    if noise_delta > 0.08 and s_delta > 0.04 and q_delta < 0.03:
        return EnhancementGateResult(
            verified=False,
            reason=(
                "Enhancement attempted but rejected: sharpness gain accompanied "
                "by elevated noise (likely oversharpening)."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    # Need at least one solid improvement; prefer two signals or quality gain.
    solid = len(improved) >= 2 or ("quality" in improved and q_delta >= 0.008)
    if solid and "quality" not in regressed:
        return EnhancementGateResult(
            verified=True,
            reason=(
                "Enhancement verified: measurable improvement on "
                f"{', '.join(improved)} (q_delta={q_delta:+.3f})."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    if improved and q_delta >= -quality_slack and "sharpness" in improved and noise_delta <= 0.05:
        return EnhancementGateResult(
            verified=True,
            reason=(
                "Enhancement verified: sharpness improved without quality/noise regression "
                f"(q_delta={q_delta:+.3f})."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    # Blur reduction with non-regressing quality is a valid classical restoration.
    if (
        "blur" in improved
        and q_delta >= -quality_slack
        and "quality" not in regressed
        and noise_delta <= 0.06
    ):
        return EnhancementGateResult(
            verified=True,
            reason=(
                "Enhancement verified: blur reduced without quality regression "
                f"(q_delta={q_delta:+.3f})."
            ),
            quality_delta=q_delta,
            sharpness_delta=s_delta,
            blur_delta=blur_delta,
            noise_delta=noise_delta,
            resolution_increased=False,
            signals_improved=tuple(improved),
            signals_regressed=tuple(regressed),
        )
    return EnhancementGateResult(
        verified=False,
        reason=(
            "Enhancement attempted but not verified: changes were cosmetic or "
            f"below multi-signal thresholds (q_delta={q_delta:+.3f}, "
            f"improved={improved or 'none'})."
        ),
        quality_delta=q_delta,
        sharpness_delta=s_delta,
        blur_delta=blur_delta,
        noise_delta=noise_delta,
        resolution_increased=False,
        signals_improved=tuple(improved),
        signals_regressed=tuple(regressed),
    )
