"""Enhancement pipeline orchestrating adaptive image improvements."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.config.enhancement_config import EnhancementConfig
from core.contracts.image_quality import ImageQualityReport
from core.logging import get_logger
from vision.enhancement.color_correction import apply_clahe, gamma_correction
from vision.enhancement.deblur import deblur
from vision.enhancement.enhancement_gate import compare_metrics
from vision.enhancement.exposure_correction import correct_exposure, enhance_low_light
from vision.enhancement.lighting import adaptive_brightness, enhance_contrast
from vision.enhancement.luminance import (
    brightness_collapsed,
    color_shifted,
    mean_luminance,
    median_luminance,
)
from vision.enhancement.noise_reduction import reduce_noise
from vision.enhancement.quality_estimator import classify_quality, measure_quality
from vision.enhancement.sharpness import reduce_jpeg_artifacts, sharpen
from vision.enhancement.super_resolution import (
    lanczos_upscale,
    unload_sr_models,
    upscale,
    validate_sr_against_baseline,
)
from vision.enhancement.white_balance import white_balance

logger = get_logger(__name__)

_CACHE_MAX = 8


class EnhancementPipeline:
    """Evaluate quality (HIGH/MEDIUM/LOW) and enhance only when needed."""

    def __init__(self, config: EnhancementConfig, models_dir: Path) -> None:
        self._config = config
        self._models_dir = models_dir
        self._cache: dict[str, tuple[NDArray[np.uint8], ImageQualityReport]] = {}

    def _cache_key(
        self,
        pixels: NDArray[np.uint8],
        *,
        competition_mode: bool,
        enable_super_resolution: bool,
    ) -> str:
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()[:16]
        # enhance_v4: multi-signal verification gate (PART 2-A).
        flags = (
            f"{competition_mode}:{enable_super_resolution}:{self._config.enabled}:"
            f"{self._config.enable_super_resolution}:{self._config.sr_scale}:enhance_v5"
        )
        return f"{digest}:{flags}"

    def process(
        self,
        pixels: NDArray[np.uint8],
        *,
        competition_mode: bool,
        enable_super_resolution: bool,
    ) -> tuple[NDArray[np.uint8], ImageQualityReport]:
        key = self._cache_key(
            pixels,
            competition_mode=competition_mode,
            enable_super_resolution=enable_super_resolution,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0].copy(), replace(cached[1])

        enhanced, report = self._process_impl(
            pixels,
            competition_mode=competition_mode,
            enable_super_resolution=enable_super_resolution,
        )
        if len(self._cache) >= _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (enhanced.copy(), report)
        return enhanced, report

    def _process_impl(
        self,
        pixels: NDArray[np.uint8],
        *,
        competition_mode: bool,
        enable_super_resolution: bool,
    ) -> tuple[NDArray[np.uint8], ImageQualityReport]:
        start = time.perf_counter()
        before = measure_quality(pixels)
        level = classify_quality(before)
        operations: list[str] = []
        current = pixels.copy()
        sr_used = False

        # HIGH: never enhance — preserve original visual content.
        # MEDIUM/LOW: enhance only when the enhancement subsystem is enabled.
        should_enhance = self._config.enabled and level in {"MEDIUM", "LOW"}
        if competition_mode and self._config.competition_always_enhance and level != "HIGH":
            should_enhance = self._config.enabled

        if not should_enhance:
            elapsed = (time.perf_counter() - start) * 1000.0
            reason = (
                "Enhancement skipped: input quality already sufficient."
                if level == "HIGH"
                else "Enhancement skipped: enhancement subsystem disabled."
            )
            logger.info(reason)
            print(
                f"[ENHANCEMENT]\n"
                f"Input: {before.resolution_width}x{before.resolution_height}\n"
                f"Quality: {level}\n"
                f"State: NOT_NEEDED\n"
                f"Reason: {reason}\n"
                f"FINAL IMAGE: ORIGINAL",
                flush=True,
            )
            report = ImageQualityReport(
                metrics=before,
                enhancement_operations=(),
                enhancement_applied=False,
                processing_time_ms=elapsed,
                improvement_percent=0.0,
                before_quality=before.estimated_quality,
                after_quality=before.estimated_quality,
                quality_level=level,
                enhancement_attempted=False,
                enhancement_rejected=False,
                rejection_reason="",
                enhancement_status="ENHANCEMENT_NOT_REQUIRED",
                enhancement_verified=False,
                verification_reason=reason,
                quality_delta_percent=0.0,
                original_width=before.resolution_width,
                original_height=before.resolution_height,
                output_width=before.resolution_width,
                output_height=before.resolution_height,
                before_sharpness=before.sharpness,
                after_sharpness=before.sharpness,
                before_blur_score=before.blur_score,
                after_blur_score=before.blur_score,
                before_noise_score=before.noise_score,
                after_noise_score=before.noise_score,
            )
            return current, report

        strong = level == "LOW"
        # Prefer clarity over tone remapping — preserve original exposure/color.
        contrast_factor = 1.12 if strong else 1.08
        sharpen_factor = 1.16 if strong else 1.08
        deblur_strength = 1.20 if strong else 1.12
        rejection_reason = ""
        orig_luma = mean_luminance(pixels)
        orig_med = median_luminance(pixels)
        min_dim_early = min(before.resolution_height, before.resolution_width)
        sr_enabled_early = bool(self._config.enable_super_resolution or enable_super_resolution)
        want_sr_early = sr_enabled_early and min_dim_early < self._config.super_resolution_min_dimension
        print(
            f"[ENHANCEMENT]\n"
            f"Input: {before.resolution_width}x{before.resolution_height}\n"
            f"Quality: {level}\n"
            f"State: RUNNING\n"
            f"SR enabled: {sr_enabled_early} | want_sr: {want_sr_early}",
            flush=True,
        )

        def _accept_op(candidate: NDArray[np.uint8], label: str, *, slack: float = 0.005) -> bool:
            """Keep an op only when it does not regress estimated quality."""
            nonlocal current
            prev_q = measure_quality(current).estimated_quality
            next_q = measure_quality(candidate).estimated_quality
            if next_q + slack >= prev_q and not brightness_collapsed(pixels, candidate):
                current = candidate
                operations.append(label)
                return True
            logger.info(
                "Enhancement op skipped after validation: %s (q %.3f → %.3f)",
                label,
                prev_q,
                next_q,
            )
            return False

        # Tone ops ONLY for true underexposure — never crush bright/normal scenes to 0.5.
        if before.brightness < self._config.min_brightness:
            current = adaptive_brightness(current, target=min(0.46, before.brightness + 0.10))
            operations.append("adaptive_brightness")
            if before.brightness < (0.20 if strong else 0.24):
                current = enhance_low_light(current)
                operations.append("low_light_enhancement")
            # Gentle exposure lift with clamped gain (luminance-based).
            if mean_luminance(current) < 0.40:
                current = correct_exposure(current, target_brightness=0.46, max_scale=1.15, min_scale=1.0)
                operations.append("exposure_correction")
        elif before.brightness > self._config.max_brightness:
            # Soft highlight pull only — do not force mid-gray.
            current = adaptive_brightness(current, target=min(before.brightness, 0.78))
            operations.append("adaptive_brightness")

        if self._config.enable_color_correction and before.white_balance_score < (0.48 if strong else 0.38):
            candidate = white_balance(current)
            _accept_op(candidate, "white_balance", slack=0.01)

        if before.contrast < self._config.min_contrast and self._config.enable_color_correction:
            current = enhance_contrast(current, factor=contrast_factor)
            operations.append("contrast_enhancement")

        # CLAHE only when contrast is badly crushed (avoid darkening mid-tones).
        if self._config.enable_clahe and before.contrast < (0.28 if strong else 0.24):
            _accept_op(apply_clahe(current), "clahe", slack=0.01)

        if self._config.enable_gamma and before.brightness < (0.30 if strong else 0.26):
            _accept_op(
                gamma_correction(current, gamma=1.12 if strong else 1.08),
                "gamma_correction",
                slack=0.01,
            )

        noise_gate = (self._config.max_noise_score * 0.85) if strong else self._config.max_noise_score
        if self._config.enable_denoise and before.noise_score > noise_gate:
            _accept_op(reduce_noise(current), "noise_reduction", slack=0.01)

        if before.blur_score > (0.48 if strong else 0.55) or before.motion_blur_score > (0.38 if strong else 0.45):
            _accept_op(deblur(current, strength=deblur_strength), "deblurring", slack=0.01)

        # JPEG cleanup can over-smooth textured / soft synthetic content — keep only if helpful.
        if before.compression_artifact_score > (0.38 if strong else 0.45):
            _accept_op(reduce_jpeg_artifacts(current), "jpeg_artifact_removal", slack=0.01)

        # Defer sharpen/clarity until after SR when SR will run — polish-before-SR
        # previously caused noise_artifacts rejection and "Not applied" with no upscale.
        deferred_sharpen = False
        if not want_sr_early and self._config.enable_sharpen and (
            before.sharpness < self._config.min_sharpness or (strong and before.sharpness < 0.35)
        ):
            candidate = sharpen(current, factor=sharpen_factor)
            if not _accept_op(candidate, "sharpening", slack=0.0):
                # Milder polish when aggressive sharpen is rejected by quality check.
                _accept_op(sharpen(current, factor=1.08), "clarity_polish", slack=0.0)
        elif want_sr_early and self._config.enable_sharpen and (
            before.sharpness < self._config.min_sharpness or (strong and before.sharpness < 0.35)
        ):
            deferred_sharpen = True

        if not want_sr_early and not operations and self._config.enable_sharpen:
            _accept_op(
                sharpen(current, factor=1.12 if strong else 1.08),
                "clarity_polish",
                slack=0.0,
            )

        # Snapshot classical result before SR so a failed SR gate can fall back.
        pre_sr_pixels = current.copy()
        pre_sr_operations = list(operations)

        min_dim = min(before.resolution_height, before.resolution_width)
        sr_meta = {
            "model": "",
            "scale": 1,
            "device": "",
            "input": f"{before.resolution_width}x{before.resolution_height}",
            "output": "",
        }
        # MEDIUM/LOW: attempt REAL super-resolution when enabled in config or UI.
        # HIGH never reaches this branch (should_enhance is false).
        sr_enabled = bool(self._config.enable_super_resolution or enable_super_resolution)
        want_sr = sr_enabled and min_dim < self._config.super_resolution_min_dimension
        sr_attempted = False
        sr_failed = False
        if want_sr:
            scale = int(getattr(self._config, "sr_scale", 2) or 2)
            # Very small inputs: still 2× by default (4× only when tiny and LOW).
            if strong and min_dim < 240:
                scale = min(4, max(2, scale))
            else:
                scale = 2
            logger.info(
                "[SR] Quality: %s | attempting real SR scale=%sx min_dim=%s",
                level,
                scale,
                min_dim,
            )
            sr_attempted = True
            sr_result = upscale(
                current,
                models_dir=self._models_dir,
                min_dimension=self._config.super_resolution_min_dimension,
                scale=scale,
                tile_size=int(getattr(self._config, "sr_tile_size", 128) or 128),
                tile_overlap=int(getattr(self._config, "sr_tile_overlap", 8) or 8),
                device=str(getattr(self._config, "sr_device", "auto") or "auto"),
                max_output_side=int(getattr(self._config, "sr_max_output_side", 2048) or 2048),
                allow_download=bool(getattr(self._config, "sr_allow_download", True)),
            )
            sr_meta["model"] = sr_result.model_name
            sr_meta["scale"] = sr_result.scale
            sr_meta["device"] = sr_result.device
            sr_meta["output"] = f"{sr_result.output_size[0]}x{sr_result.output_size[1]}"
            if sr_result.true_sr:
                print(
                    f"[SR DEBUG]\nValidation started: TRUE\n"
                    f"Candidate size: {sr_result.output_size[0]}x{sr_result.output_size[1]}",
                    flush=True,
                )
                ok, reason = validate_sr_against_baseline(
                    pixels if np.array_equal(current, pixels) else current,
                    sr_result.pixels,
                    scale=sr_result.scale,
                )
                print(
                    f"[SR DEBUG]\nValidation result: {'PASS' if ok else 'FAIL'}\n"
                    f"Validation reason: {reason}",
                    flush=True,
                )
                if ok:
                    current = sr_result.pixels
                    operations.append(f"super_resolution:{sr_result.backend}:{sr_result.model_name}")
                    sr_used = True
                    try:
                        from PIL import Image as _PilDbg

                        dbg = Path("tmp") / "debug_super_resolution_output.png"
                        dbg.parent.mkdir(parents=True, exist_ok=True)
                        _PilDbg.fromarray(current).save(dbg)
                        print(f"[SR DEBUG]\nSaved debug output: {dbg.resolve()}", flush=True)
                    except Exception as dbg_exc:  # noqa: BLE001
                        print(f"[SR DEBUG]\nFailed to save debug output: {dbg_exc}", flush=True)
                    logger.info("[SR] Validation: PASSED (%s) | Final image: ENHANCED", reason)
                else:
                    sr_failed = True
                    logger.info(
                        "[SR] Validation: FAILED | Reason: %s | Final image: ORIGINAL",
                        reason,
                    )
                    if not rejection_reason:
                        rejection_reason = f"Enhanced image did not pass quality validation ({reason})"
            else:
                sr_failed = True
                logger.info(
                    "[SR] Validation: FAILED | Reason: %s | Final image: ORIGINAL",
                    sr_result.message or sr_result.backend,
                )
                if not rejection_reason:
                    rejection_reason = (
                        "Real super-resolution unavailable "
                        f"({sr_result.message or 'no model weights'})"
                        if sr_result.backend == "failed"
                        else f"Super-resolution did not run ({sr_result.message or sr_result.backend})"
                    )
            # Free SR weights before downstream vision models load.
            unload_sr_models()

        # When SR did not stick, finish deferred classical sharpen on the pre-SR result.
        if want_sr and not sr_used:
            current = pre_sr_pixels
            operations = list(pre_sr_operations)
            if deferred_sharpen or (
                self._config.enable_sharpen
                and before.sharpness < self._config.min_sharpness
                and "sharpening" not in operations
                and "clarity_polish" not in operations
            ):
                factor = sharpen_factor if deferred_sharpen or strong else 1.10
                current = sharpen(current, factor=factor)
                operations.append("sharpening" if deferred_sharpen else "clarity_polish")
            # SR-only rejection must not block classical clarity fallback below.
            if rejection_reason and not rejection_reason.startswith("Enhancement rejected:"):
                rejection_reason = ""
            after = measure_quality(current)
        else:
            after = measure_quality(current)
        baseline_metrics = before
        # When true SR changed resolution, compare fairly against Lanczos at the
        # same output size (do not mix native-res scores with 2× scores).
        if sr_used and current.shape[:2] != pixels.shape[:2]:
            baseline = lanczos_upscale(pixels, max(1, int(sr_meta["scale"] or 2)))
            if baseline.shape != current.shape:
                from PIL import Image as _PilImage

                baseline = np.asarray(
                    _PilImage.fromarray(baseline).resize(
                        (current.shape[1], current.shape[0]),
                        _PilImage.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
            baseline_metrics = measure_quality(baseline)
            # Optional light polish on SR output — never used as primary enhancer.
            if (
                operations
                and after.estimated_quality < self._config.quality_threshold
                and after.estimated_quality + 0.01 >= baseline_metrics.estimated_quality
            ):
                refined = current
                if after.sharpness < self._config.min_sharpness and self._config.enable_sharpen:
                    refined = sharpen(refined, factor=1.10)
                    operations.append("adaptive_sharpen_pass2")
                candidate = measure_quality(refined)
                if candidate.estimated_quality >= after.estimated_quality:
                    current = refined
                    after = candidate
        else:
            # Second light pass toward quality target — only if still weak and improving.
            if (
                operations
                and after.estimated_quality < self._config.quality_threshold
                and after.estimated_quality >= before.estimated_quality
            ):
                refined = current
                if after.sharpness < self._config.min_sharpness and self._config.enable_sharpen:
                    refined = sharpen(refined, factor=1.15 if strong else 1.1)
                    operations.append("adaptive_sharpen_pass2")
                if after.contrast < self._config.min_contrast and self._config.enable_color_correction:
                    refined = enhance_contrast(refined, factor=1.12 if strong else 1.08)
                    operations.append("adaptive_contrast_pass2")
                candidate = measure_quality(refined)
                if candidate.estimated_quality >= after.estimated_quality:
                    current = refined
                    after = candidate

            # Hard safety: luma/chroma collapse always reverts before gate.
            if operations and brightness_collapsed(pixels, current):
                current = pixels.copy()
                after = before
                operations = []
                sr_used = False
                rejection_reason = (
                    f"Enhancement rejected: darkened image "
                    f"(luma {orig_luma:.3f} → {mean_luminance(current):.3f})."
                )
            elif operations and (
                color_shifted(pixels, current) or self._shifted_toward_gray(pixels, current)
            ):
                current = pixels.copy()
                after = before
                operations = []
                sr_used = False
                rejection_reason = "Enhancement rejected: color/chroma shifted unacceptably."

        # If heavier ops were rejected, still try a single safe clarity polish.
        if (
            not operations
            and strong
            and self._config.enable_sharpen
            and not sr_used
            and not rejection_reason.startswith("Enhancement rejected: darkened")
        ):
            polished = sharpen(pixels.copy(), factor=1.18)
            polished_metrics = measure_quality(polished)
            if (
                polished_metrics.estimated_quality + 0.005 >= before.estimated_quality
                and polished_metrics.sharpness + 0.01 >= before.sharpness
                and polished_metrics.blur_score <= before.blur_score + 0.03
                and not brightness_collapsed(pixels, polished)
                and not self._shifted_toward_gray(pixels, polished)
            ):
                current = polished
                after = polished_metrics
                operations = ["clarity_polish"]
        elif (
            not operations
            and self._config.enable_sharpen
            and before.sharpness < self._config.min_sharpness
            and not sr_used
            and not rejection_reason.startswith("Enhancement rejected:")
        ):
            polished = sharpen(pixels.copy(), factor=1.08)
            polished_metrics = measure_quality(polished)
            if (
                polished_metrics.estimated_quality + 0.005 >= before.estimated_quality
                and polished_metrics.sharpness >= before.sharpness
                and polished_metrics.blur_score <= before.blur_score + 0.02
                and not brightness_collapsed(pixels, polished)
                and not self._shifted_toward_gray(pixels, polished)
            ):
                current = polished
                after = polished_metrics
                operations = ["clarity_polish"]

        # Do not claim enhancement if pixels are effectively unchanged.
        if operations and current.shape == pixels.shape and np.array_equal(current, pixels):
            operations = []
            after = before
            rejection_reason = rejection_reason or (
                "Enhancement attempted but not verified: no visible pixel change."
            )

        # --- Multi-signal verification gate (PART 2-A) ---
        gate = None
        verified = False
        verification_reason = rejection_reason
        if operations:
            q_slack = 0.025 if strong else 0.01
            s_slack = 0.035 if strong else 0.02
            gate = compare_metrics(
                before,
                after,
                baseline=baseline_metrics if sr_used else None,
                is_super_resolution=sr_used,
                quality_slack=q_slack,
                sharpness_slack=s_slack,
            )
            verified = bool(gate.verified)
            verification_reason = gate.reason
            if not verified:
                if sr_used:
                    # SR gate failed after upscale — fall back to classical pre-SR result
                    # instead of discarding all improvement and keeping a raw original.
                    current = pre_sr_pixels.copy()
                    operations = list(pre_sr_operations)
                    sr_used = False
                    if deferred_sharpen or (
                        self._config.enable_sharpen
                        and before.sharpness < self._config.min_sharpness
                        and "sharpening" not in operations
                        and "clarity_polish" not in operations
                    ):
                        current = sharpen(
                            current,
                            factor=sharpen_factor if (deferred_sharpen or strong) else 1.10,
                        )
                        operations.append("sharpening" if deferred_sharpen else "clarity_polish")
                    after = measure_quality(current)
                    if operations and not (
                        current.shape == pixels.shape and np.array_equal(current, pixels)
                    ):
                        classical = compare_metrics(
                            before,
                            after,
                            baseline=None,
                            is_super_resolution=False,
                            quality_slack=q_slack,
                            sharpness_slack=s_slack,
                        )
                        if classical.verified:
                            gate = classical
                            verified = True
                            verification_reason = (
                                "Super-resolution not confirmed; classical enhancement verified: "
                                + classical.reason
                            )
                            rejection_reason = ""
                        else:
                            current = pixels.copy()
                            after = before
                            operations = []
                            rejection_reason = gate.reason
                            verification_reason = gate.reason
                    else:
                        current = pixels.copy()
                        after = before
                        operations = []
                        rejection_reason = gate.reason
                else:
                    # Never replace original with an unverified / worse candidate.
                    current = pixels.copy()
                    after = before
                    operations = []
                    sr_used = False
                    rejection_reason = gate.reason

        quality_delta = (
            float(gate.quality_delta)
            if gate is not None
            else float(after.estimated_quality - before.estimated_quality)
        )
        # Display improvement only when verified; signed delta always recorded.
        improvement = max(0.0, quality_delta * 100.0) if verified else 0.0
        applied = bool(verified)

        # Status mapping — three user-facing states.
        if sr_attempted and sr_failed and not applied and not operations:
            if "unavailable" in (rejection_reason or "").lower() or "did not run" in (
                rejection_reason or ""
            ).lower():
                enhancement_status = "ENHANCEMENT_FAILED"
            else:
                enhancement_status = "ENHANCEMENT_ATTEMPTED_UNVERIFIED"
            final_label = "ORIGINAL"
        elif applied and verified:
            enhancement_status = "ENHANCEMENT_VERIFIED"
            final_label = "ENHANCED"
            rejection_reason = ""
        elif rejection_reason and not applied:
            # Distinguish hard reject vs soft unverified.
            if "rejected" in rejection_reason.lower() or "regressed" in rejection_reason.lower():
                enhancement_status = "ENHANCEMENT_REJECTED"
            else:
                enhancement_status = "ENHANCEMENT_ATTEMPTED_UNVERIFIED"
            final_label = "ORIGINAL"
        elif not applied:
            enhancement_status = "ENHANCEMENT_ATTEMPTED_UNVERIFIED"
            final_label = "ORIGINAL"
            if not verification_reason:
                verification_reason = (
                    "Enhancement attempted but not verified: no measurable improvement."
                )
        else:
            enhancement_status = "ENHANCEMENT_VERIFIED"
            final_label = "ENHANCED"

        # Compat alias: UI historically keys on ENHANCEMENT_APPLIED for success.
        if enhancement_status == "ENHANCEMENT_VERIFIED":
            status_for_report = "ENHANCEMENT_APPLIED"
        else:
            status_for_report = enhancement_status

        final_luma = mean_luminance(current)
        logger.info(
            "Enhancement decision level=%s attempted=True verified=%s applied=%s "
            "status=%s q_delta=%+.3f luma_orig=%.3f luma_final=%.3f reason=%s ops=%s",
            level,
            verified,
            applied,
            status_for_report,
            quality_delta,
            orig_luma,
            final_luma,
            verification_reason or rejection_reason or "-",
            list(operations),
        )
        print(
            f"[ENHANCEMENT]\n"
            f"Original: {before.resolution_width}x{before.resolution_height}\n"
            f"Output: {after.resolution_width}x{after.resolution_height}\n"
            f"Quality level: {level}\n"
            f"Attempted: TRUE | Verified: {verified} | Applied: {applied}\n"
            f"Status: {status_for_report}\n"
            f"Reason: {verification_reason or rejection_reason or '-'}\n"
            f"q_delta={quality_delta:+.3f} sharp={before.sharpness:.3f}->{after.sharpness:.3f}\n"
            f"FINAL IMAGE: {final_label}",
            flush=True,
        )

        elapsed = (time.perf_counter() - start) * 1000.0
        report = ImageQualityReport(
            metrics=replace(after),
            enhancement_operations=tuple(operations),
            enhancement_applied=applied,
            processing_time_ms=elapsed,
            improvement_percent=improvement,
            before_quality=before.estimated_quality,
            after_quality=after.estimated_quality,
            super_resolution_used=sr_used,
            quality_level=level,
            enhancement_attempted=True,
            enhancement_rejected=status_for_report == "ENHANCEMENT_REJECTED",
            rejection_reason=(rejection_reason or verification_reason) if not applied else "",
            enhancement_status=status_for_report,
            sr_model=sr_meta["model"] if (sr_used or sr_attempted) else "",
            sr_scale=int(sr_meta["scale"] or 1) if (sr_used or sr_attempted) else 1,
            sr_device=sr_meta["device"] if (sr_used or sr_attempted) else "",
            sr_input_size=sr_meta["input"] if (sr_used or sr_attempted) else "",
            sr_output_size=sr_meta["output"] if sr_used else "",
            enhancement_verified=verified,
            verification_reason=verification_reason
            if verified
            else (rejection_reason or verification_reason),
            quality_delta_percent=quality_delta * 100.0,
            original_width=before.resolution_width,
            original_height=before.resolution_height,
            output_width=after.resolution_width if applied else before.resolution_width,
            output_height=after.resolution_height if applied else before.resolution_height,
            before_sharpness=before.sharpness,
            after_sharpness=after.sharpness if applied else before.sharpness,
            before_blur_score=before.blur_score,
            after_blur_score=after.blur_score if applied else before.blur_score,
            before_noise_score=before.noise_score,
            after_noise_score=after.noise_score if applied else before.noise_score,
        )
        return current, report

    @staticmethod
    def _shifted_toward_gray(original: NDArray[np.uint8], enhanced: NDArray[np.uint8]) -> bool:
        """Detect harmful gray-world style desaturation from enhancement ops."""
        if original.shape != enhanced.shape or original.ndim != 3:
            return False
        orig = original.reshape(-1, 3).astype(np.float32)
        enh = enhanced.reshape(-1, 3).astype(np.float32)
        if orig.shape[0] < 16:
            return False
        orig_chroma = float(np.mean(np.std(orig, axis=1)))
        enh_chroma = float(np.mean(np.std(enh, axis=1)))
        if orig_chroma < 8.0:
            return False
        return enh_chroma < orig_chroma * 0.72
