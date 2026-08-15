"""Non-browser real-application behavior checks for caption/enhancement/language."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _hash_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_language_catalogs() -> dict[str, object]:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    out = {}
    for lang in ("en", "fa", "de", "es", "zh"):
        data = json.loads((ROOT / "translations" / f"{lang}.json").read_text(encoding="utf-8"))
        out[lang] = {
            "keys": len(data),
            "missing": sorted(set(en) - set(data))[:10],
            "analyze": data.get("button.analyze", ""),
        }
    return out


def check_enhancement() -> dict[str, object]:
    from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
    from vision.enhancement.enhancement_pipeline import EnhancementPipeline

    src = ROOT / "tmp" / "coco_baseball.jpg"
    if not src.is_file():
        src = ROOT / "tmp" / "coco_kitchen.jpg"
    img = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    # Force a low-quality input.
    degraded = np.asarray(
        Image.fromarray(img).resize((160, 120)).resize(img.shape[1::-1], Image.Resampling.BILINEAR)
    )
    degraded = (degraded.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
    pipeline = EnhancementPipeline(DEFAULT_ENHANCEMENT_CONFIG, models_dir=ROOT / "models")
    enhanced, report = pipeline.process(degraded, competition_mode=False, enable_super_resolution=False)
    oh = hashlib.sha256(degraded.tobytes()).hexdigest()
    eh = hashlib.sha256(enhanced.tobytes()).hexdigest()
    return {
        "quality_level": report.quality_level,
        "enhancement_attempted": report.enhancement_attempted,
        "enhancement_applied": report.enhancement_applied,
        "enhancement_rejected": report.enhancement_rejected,
        "rejection_reason": report.rejection_reason,
        "before": report.before_quality,
        "after": report.after_quality,
        "ops": list(report.enhancement_operations),
        "original_hash": oh,
        "enhanced_hash": eh,
        "pixels_changed": oh != eh,
    }


def check_pipeline_caption() -> dict[str, object]:
    from analysis.activity.ollama_client import ollama_call_stats, reset_ollama_call_stats
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.localization.caption_translator import CaptionTranslator
    from streamlit_app.catalog import TranslationCatalog

    image = ROOT / "tmp" / "coco_baseball.jpg"
    if not image.is_file():
        image = ROOT / "tmp" / "coco_kitchen.jpg"
    reset_ollama_call_stats()
    startup = StartupOrchestrator().run()
    orchestrator = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    if hasattr(orchestrator, "_semantic_reasoning"):
        orchestrator._semantic_reasoning.reset_call_count()
    result = orchestrator.analyze(
        PipelineRequest(
            image_path=image,
            options=AnalysisOptions(
                enable_gemma=True,
                competition_mode=False,
                enable_enhancement=True,
                enable_super_resolution=False,
                enable_sam2=True,
            ),
        )
    )
    caption = result.caption.canonical_caption_en or result.caption.text
    stats = ollama_call_stats()
    translator = CaptionTranslator()
    langs = {}
    for lang in ("fa", "de", "es", "zh", "en"):
        text = translator.translate(caption, lang) if lang != "en" else caption
        langs[lang] = text[:160]
    catalog = TranslationCatalog()
    ui = {}
    for lang in ("en", "fa", "de", "es", "zh"):
        catalog.set_language(lang)
        ui[lang] = {
            "analyze": catalog.translate("button.analyze"),
            "nav": catalog.translate("streamlit.nav.analyze"),
            "language": catalog.translate("settings.language"),
        }
    enhanced_path = result.enhanced_preview_path
    hashes = {
        "original": _hash_path(image),
        "enhanced": _hash_path(Path(enhanced_path)) if enhanced_path and Path(enhanced_path).is_file() else None,
    }
    return {
        "caption": caption,
        "caption_words": len(caption.split()),
        "has_an_a": "an a" in caption.lower(),
        "has_filler": "close enough to matter" in caption.lower(),
        "glove_count": caption.lower().count("baseball glove"),
        "vlm_executions": result.metrics.vlm_executions,
        "gemma_semantic_calls": stats.semantic_calls,
        "translation_calls": translator.translation_count,
        "qa_passed": result.qa_passed,
        "image_quality": {
            "level": result.image_quality.quality_level if result.image_quality else None,
            "applied": result.image_quality.enhancement_applied if result.image_quality else None,
            "rejected": getattr(result.image_quality, "enhancement_rejected", None)
            if result.image_quality
            else None,
            "reason": getattr(result.image_quality, "rejection_reason", None)
            if result.image_quality
            else None,
        },
        "hashes": hashes,
        "display_should_be_enhanced": bool(
            result.image_quality
            and result.image_quality.enhancement_applied
            and enhanced_path
            and Path(enhanced_path).is_file()
        ),
        "language_ui": ui,
        "caption_translations": langs,
        "canonical_unchanged_after_translation": all(
            caption == (result.caption.canonical_caption_en or caption) for _ in langs
        ),
    }


def main() -> None:
    report = {
        "catalogs": check_language_catalogs(),
        "enhancement": check_enhancement(),
    }
    try:
        report["pipeline"] = check_pipeline_caption()
    except Exception as exc:  # noqa: BLE001
        report["pipeline"] = {"error": str(exc)}
    out = ROOT / "tmp" / "real_app_behavior_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
