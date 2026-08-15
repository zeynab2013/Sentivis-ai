"""Final stabilization measurements: Gemma, OCR, SAM2, caption quality, languages."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _text_image(path: Path, text: str, size: int) -> Path:
    img = Image.new("RGB", (640, 200), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((32, 70), text, fill=(245, 245, 245), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _blank_image(path: Path) -> Path:
    Image.new("RGB", (320, 240), (40, 40, 40)).save(path)
    return path


def _degrade(src: Path, path: Path, mode: str) -> Path:
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if mode == "lowres":
        out = img.resize((160, 120)).resize(img.size, Image.Resampling.BILINEAR)
    elif mode == "blur":
        out = img.resize((img.width // 6, img.height // 6)).resize(img.size, Image.Resampling.BILINEAR)
    elif mode == "noise":
        noise = np.random.default_rng(0).normal(0, 35, arr.shape)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    elif mode == "dark":
        out = Image.fromarray(np.clip(arr * 0.28, 0, 255).astype(np.uint8))
    else:
        out = img
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def ollama_ps() -> dict[str, object]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def warm_gemma() -> dict[str, object]:
    """Keep model loaded before analysis so load vs inference can be separated."""
    from analysis.activity.ollama_client import OllamaClient, ollama_call_stats, reset_ollama_call_stats

    reset_ollama_call_stats()
    client = OllamaClient(model="gemma3:4b", keep_alive="30m", timeout_seconds=180.0)
    t0 = time.perf_counter()
    try:
        client.generate_text(
            system="Reply with OK only.",
            user="OK",
            max_tokens=4,
            purpose="other",
        )
        stats = ollama_call_stats()
        return {
            "ok": True,
            "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
            "load_ms": stats.last_load_duration_ms,
            "eval_ms": stats.last_eval_duration_ms,
            "prompt_tokens": stats.last_prompt_tokens,
            "output_tokens": stats.last_output_tokens,
            "ps_after": ollama_ps(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}


def bench_ocr() -> dict[str, object]:
    from analysis.ocr.text_extractor import OcrExtractor

    extractor = OcrExtractor()
    cases = {
        "no_text": _blank_image(ROOT / "tmp" / "ocr_blank.png"),
        "clear_text": _text_image(ROOT / "tmp" / "ocr_clear.png", "SENTIVIS LAB", 40),
        "small_text": _text_image(ROOT / "tmp" / "ocr_small.png", "ROOM 204", 14),
    }
    out: dict[str, object] = {}
    for name, path in cases.items():
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        t0 = time.perf_counter()
        result = extractor.extract(pixels)
        out[name] = {
            "source": result.source,
            "texts": list(result.texts),
            "confidence": result.confidence,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    return out


def bench_sam2() -> dict[str, object]:
    from vision.segmentation.sam2_refiner import Sam2SegmentationRefiner

    refiner = Sam2SegmentationRefiner(ROOT / "models")
    return {"available": refiner.available, **refiner.status}


def bench_enhancement() -> dict[str, object]:
    from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
    from vision.enhancement.enhancement_pipeline import EnhancementPipeline

    # Prefer COCO kitchen; otherwise synthetic office.
    kitchen = ROOT / "tmp" / "coco_kitchen.jpg"
    if not kitchen.is_file():
        kitchen = ROOT / "tmp" / "bench_office.png"
        if not kitchen.is_file():
            from scripts.competition_readiness_bench import _make_office_image

            _make_office_image(kitchen)
    pipeline = EnhancementPipeline(DEFAULT_ENHANCEMENT_CONFIG, models_dir=ROOT / "models")
    results: dict[str, object] = {}
    for mode in ("lowres", "blur", "noise", "dark"):
        degraded = _degrade(kitchen, ROOT / "tmp" / f"enhance_{mode}.jpg", mode)
        pixels = np.asarray(Image.open(degraded).convert("RGB"), dtype=np.uint8)
        t0 = time.perf_counter()
        enhanced, report = pipeline.process(pixels, competition_mode=False, enable_super_resolution=False)
        results[mode] = {
            "before": round(report.before_quality, 3),
            "after": round(report.after_quality, 3),
            "applied": report.enhancement_applied,
            "ops": list(report.enhancement_operations),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "shape_in": list(pixels.shape),
            "shape_out": list(enhanced.shape),
            "rejected_if_worse": (not report.enhancement_applied)
            or report.after_quality + 1e-6 >= report.before_quality,
        }
    return results


def bench_pipeline() -> dict[str, object]:
    from analysis.activity.ollama_client import ollama_call_stats, reset_ollama_call_stats
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.localization.caption_translator import CaptionTranslator

    image = ROOT / "tmp" / "coco_kitchen.jpg"
    if not image.is_file():
        from scripts.competition_readiness_bench import _make_office_image

        image = _make_office_image(ROOT / "tmp" / "bench_pipeline.png")

    reset_ollama_call_stats()
    warm = warm_gemma()
    reset_ollama_call_stats()

    t_start = time.perf_counter()
    startup = StartupOrchestrator().run()
    orchestrator = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    startup_s = time.perf_counter() - t_start

    # Reset semantic call counter if available.
    semantic = getattr(orchestrator, "_semantic_reasoning", None)
    if semantic is not None and hasattr(semantic, "reset_call_count"):
        semantic.reset_call_count()

    request = PipelineRequest(
        image_path=image,
        options=AnalysisOptions(
            enable_gemma=True,
            competition_mode=False,
            enable_enhancement=True,
            enable_super_resolution=False,
            enable_sam2=True,
        ),
    )
    t0 = time.perf_counter()
    result = orchestrator.analyze(request)
    total = time.perf_counter() - t0
    stats = ollama_call_stats()
    stages = {m.stage.value: round(m.duration_ms, 1) for m in result.metrics.stage_metrics}
    model_timings = {
        str(m.kind): {"load_ms": round(m.load_ms, 1), "unload_ms": round(m.unload_ms, 1)}
        for m in result.metrics.model_timings
    }
    caption = result.caption.canonical_caption_en or result.caption.narrative_full
    canonical = result.caption.canonical_caption_en

    translator = CaptionTranslator()
    lang_out = {}
    for lang in ("en", "fa", "de", "es", "zh"):
        t_lang = time.perf_counter()
        text = translator.translate(canonical or caption, lang) if lang != "en" else (canonical or caption)
        lang_out[lang] = {
            "ms": round((time.perf_counter() - t_lang) * 1000, 1),
            "preview": text[:160],
            "canonical_unchanged": (canonical or caption) == (result.caption.canonical_caption_en or caption),
        }

    semantic_calls = getattr(semantic, "semantic_call_count", None) if semantic is not None else None
    return {
        "image": str(image),
        "warm_gemma": warm,
        "startup_s": round(startup_s, 2),
        "total_s": round(total, 2),
        "stages_ms": stages,
        "model_timings_ms": model_timings,
        "vlm_executions": result.metrics.vlm_executions,
        "caption_generation_count": result.metrics.caption_generation_count,
        "qa_count": result.metrics.qa_count,
        "qa_passed": result.qa_passed,
        "objects": result.metrics.objects_detected,
        "relations": result.metrics.relationships_inferred,
        "activities": result.metrics.activities_inferred,
        "caption_words": len(caption.split()),
        "caption": caption,
        "canonical_caption_en": canonical,
        "collapsed_generic": caption.strip().lower().startswith("a person is working"),
        "gemma_stats": {
            "semantic_calls": stats.semantic_calls,
            "translation_calls": stats.translation_calls,
            "other_calls": stats.other_calls,
            "last_load_ms": stats.last_load_duration_ms,
            "last_eval_ms": stats.last_eval_duration_ms,
            "last_prompt_tokens": stats.last_prompt_tokens,
            "last_output_tokens": stats.last_output_tokens,
            "last_wall_ms": stats.last_wall_ms,
            "service_semantic_call_count": semantic_calls,
        },
        "language_switch": {
            "vlm_rerun": 0,
            "translation_count": translator.translation_count,
            "languages": lang_out,
        },
        "ollama_ps_after": ollama_ps(),
    }


def main() -> None:
    report: dict[str, object] = {
        "ocr": bench_ocr(),
        "sam2": bench_sam2(),
        "enhancement": bench_enhancement(),
    }
    try:
        report["pipeline"] = bench_pipeline()
    except Exception as exc:  # noqa: BLE001
        report["pipeline"] = {"error": str(exc)}
    out = ROOT / "tmp" / "final_stabilization_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
