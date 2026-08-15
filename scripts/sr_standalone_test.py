"""Minimal standalone Real-ESRGAN test: image → SR → output. No caption/UI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from vision.enhancement.super_resolution import unload_sr_models, upscale


def main() -> int:
    candidates = [
        ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg",
        *sorted((ROOT / "tmp" / "uploads").glob("*.jpg")),
        *sorted((ROOT / "tmp" / "uploads").glob("*.png")),
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        # Synthesize a 500×333 test image if none uploaded.
        src = ROOT / "tmp" / "sr_test_500x333.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        arr = rng.integers(30, 220, size=(333, 500, 3), dtype=np.uint8)
        Image.fromarray(arr).save(src)

    pixels = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    h, w = pixels.shape[:2]
    out_path = ROOT / "tmp" / "debug_super_resolution_output.png"
    models_dir = ROOT / "models"

    print("=== STANDALONE SR TEST ===")
    print(f"input image path: {src}")
    print(f"input dimensions: {w}x{h}")
    print(f"models_dir: {models_dir}")

    t0 = time.perf_counter()
    try:
        result = upscale(
            pixels,
            models_dir=models_dir,
            min_dimension=720,
            scale=2,
            tile_size=128,
            tile_overlap=8,
            device="auto",
            max_output_side=2048,
            allow_download=True,
        )
        elapsed = time.perf_counter() - t0
        Image.fromarray(result.pixels).save(out_path)
        print(f"model: {result.model_name or '(none)'}")
        print(f"backend: {result.backend}")
        print(f"device: {result.device}")
        print(f"true_sr: {result.true_sr}")
        print(f"message: {result.message}")
        print(f"output dimensions: {result.output_size[0]}x{result.output_size[1]}")
        print(f"output image path: {out_path}")
        print(f"execution time: {elapsed:.2f}s")
        unload_sr_models()
        if not result.true_sr:
            print("FAIL: true_sr is False")
            return 1
        if result.output_size[0] <= w or result.output_size[1] <= h:
            print("FAIL: output did not increase resolution")
            return 1
        print("PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"execution time: {elapsed:.2f}s")
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
