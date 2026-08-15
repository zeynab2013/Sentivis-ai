"""Run real super-resolution on the farm LOW-quality image and report metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE = ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg"
OUT_DIR = ROOT / "tmp" / "sr_validation"
OUT_JSON = ROOT / "tmp" / "farm_sr_probe.json"


def main() -> int:
    from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
    from vision.enhancement.enhancement_pipeline import EnhancementPipeline
    from vision.enhancement.super_resolution import lanczos_upscale, laplacian_variance

    if not IMAGE.is_file():
        raise SystemExit(f"missing {IMAGE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(Image.open(IMAGE).convert("RGB"), dtype=np.uint8)
    models_dir = ROOT / "models"
    pipeline = EnhancementPipeline(DEFAULT_ENHANCEMENT_CONFIG, models_dir=models_dir)

    enhanced, report = pipeline.process(
        pixels,
        competition_mode=False,
        enable_super_resolution=True,
    )

    Image.fromarray(pixels).save(OUT_DIR / "original.png")
    Image.fromarray(enhanced).save(OUT_DIR / "enhanced.png")
    if enhanced.shape != pixels.shape:
        baseline = lanczos_upscale(pixels, max(1, report.sr_scale or 2))
        Image.fromarray(baseline).save(OUT_DIR / "lanczos_baseline.png")
        sharp_sr = laplacian_variance(enhanced)
        sharp_base = laplacian_variance(baseline)
    else:
        sharp_sr = laplacian_variance(enhanced)
        sharp_base = laplacian_variance(pixels)

    payload = {
        "input": f"{pixels.shape[1]}x{pixels.shape[0]}",
        "output": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
        "quality_level": report.quality_level,
        "enhancement_applied": report.enhancement_applied,
        "super_resolution_used": report.super_resolution_used,
        "sr_model": report.sr_model,
        "sr_scale": report.sr_scale,
        "sr_device": report.sr_device,
        "before_quality": round(report.before_quality, 4),
        "after_quality": round(report.after_quality, 4),
        "improvement_percent": round(report.improvement_percent, 3),
        "operations": list(report.enhancement_operations),
        "rejection_reason": report.rejection_reason,
        "laplacian_sr": round(sharp_sr, 2),
        "laplacian_baseline": round(sharp_base, 2),
        "shapes_differ": enhanced.shape != pixels.shape,
        "pixel_diff_mean": float(
            np.mean(np.abs(enhanced.astype(np.float32) - pixels.astype(np.float32)))
            if enhanced.shape == pixels.shape
            else -1.0
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Wrote", OUT_JSON)
    print("Images:", OUT_DIR)
    return 0 if report.super_resolution_used and report.enhancement_applied else 1


if __name__ == "__main__":
    raise SystemExit(main())
