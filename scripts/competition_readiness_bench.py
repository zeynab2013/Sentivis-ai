"""Competition-readiness benchmarks: enhancement, OCR, translation, optional pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _make_office_image(path: Path) -> Path:
    img = Image.new("RGB", (960, 640), (48, 56, 72))
    draw = ImageDraw.Draw(img)
    # desk
    draw.rectangle((120, 360, 840, 520), fill=(96, 78, 58))
    # monitor
    draw.rectangle((360, 180, 620, 340), fill=(30, 30, 35))
    draw.rectangle((380, 200, 600, 320), fill=(70, 140, 200))
    # keyboard
    draw.rectangle((400, 400, 580, 430), fill=(40, 40, 45))
    # chair
    draw.ellipse((430, 500, 550, 600), fill=(20, 20, 24))
    # person torso
    draw.rectangle((450, 300, 530, 420), fill=(25, 45, 90))
    draw.ellipse((460, 240, 520, 300), fill=(210, 180, 150))
    # phone
    draw.rectangle((650, 390, 700, 420), fill=(60, 60, 70))
    # OCR text on monitor bezel / poster
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((140, 80), "SENTIVIS LAB", fill=(240, 240, 240), font=font)
    draw.text((140, 120), "ROOM 204", fill=(220, 220, 220), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _make_blurry(path: Path, src: Path) -> Path:
    img = Image.open(src).resize((240, 160)).resize((960, 640), Image.Resampling.BILINEAR)
    img.save(path)
    return path


def bench_enhancement() -> dict[str, object]:
    from core.config.defaults import DEFAULT_ENHANCEMENT_CONFIG
    from vision.enhancement.enhancement_pipeline import EnhancementPipeline

    pipeline = EnhancementPipeline(DEFAULT_ENHANCEMENT_CONFIG, models_dir=ROOT / "models")
    office = _make_office_image(ROOT / "tmp" / "bench_office.png")
    blurry = _make_blurry(ROOT / "tmp" / "bench_blurry.png", office)
    results = {}
    for label, path in (("high_office", office), ("blurry", blurry)):
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        t0 = time.perf_counter()
        enhanced, report = pipeline.process(
            pixels, competition_mode=False, enable_super_resolution=True
        )
        dt = (time.perf_counter() - t0) * 1000
        results[label] = {
            "quality_level": report.quality_level,
            "before": round(report.before_quality, 3),
            "after": round(report.after_quality, 3),
            "applied": report.enhancement_applied,
            "ops": list(report.enhancement_operations),
            "ms": round(dt, 1),
            "rejected_worse": report.enhancement_applied is False
            and report.after_quality + 0.01 < report.before_quality,
            "shape": list(enhanced.shape),
        }
    return results


def bench_ocr() -> dict[str, object]:
    from analysis.ocr.text_extractor import OcrExtractor

    path = _make_office_image(ROOT / "tmp" / "bench_ocr.png")
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    t0 = time.perf_counter()
    result = OcrExtractor().extract(pixels)
    return {
        "source": result.source,
        "texts": list(result.texts),
        "confidence": result.confidence,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "working": bool(result.texts) and result.source != "none",
    }


def bench_translation() -> dict[str, object]:
    from language.localization.caption_translator import CaptionTranslator

    canonical = (
        "A person in a navy jacket sits at a desk in an office workspace, "
        "with a computer monitor, keyboard, and telephone visible on the desk."
    )
    translator = CaptionTranslator()
    out: dict[str, object] = {"canonical_words": len(canonical.split())}
    for lang in ("fa", "de", "es", "zh"):
        t0 = time.perf_counter()
        text = translator.translate(canonical, lang)
        out[lang] = {
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "preview": text[:160],
            "words": len(text.split()) if lang != "zh" else len(text),
            "changed": text.strip() != canonical.strip(),
        }
    out["translation_count"] = translator.translation_count
    return out


def bench_metrics() -> dict[str, object]:
    from core.contracts.analysis import (
        ActivityEvidence,
        ActivityHints,
        AttributeSet,
        EnvironmentInfo,
        Relation,
        SceneContext,
        SceneGraph,
        SceneNode,
    )
    from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator

    env = EnvironmentInfo(
        scene_type="office",
        setting="office",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="indoor",
        social_context="none",
        crowd_level="empty",
        scene_complexity="medium",
        evidence=(),
    )
    empty = SceneContext(
        graph=SceneGraph(nodes=(), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=env,
        object_count=0,
        dominant_objects=(),
        spatial_summary="",
    )
    with_rel = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "a", "person", 0.9, "m"),
                SceneNode(1, "b", "desk", 0.9, "m"),
            ),
            relations=(Relation(0, 1, "sitting_on", 0.9),),
        ),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    activity="working at computer",
                    confidence=0.9,
                    supporting_node_indices=(0,),
                    supporting_relation_types=("sitting_on",),
                    rationale="desk and monitor",
                ),
            ),
            confidence=0.9,
        ),
        environment=env,
        object_count=2,
        dominant_objects=("person", "desk"),
        spatial_summary="",
    )
    ev = CaptionQualityEvaluator()
    r0 = ev.evaluate("A scene.", empty)
    r1 = ev.evaluate("A person is sitting on a desk while working at computer.", with_rel)
    r2 = ev.evaluate("A person stands alone.", with_rel)
    return {
        "empty_object": r0.object_coverage,
        "empty_relation": r0.relationship_coverage,
        "empty_activity": r0.activity_coverage,
        "empty_hall": r0.hallucination_risk,
        "covered_relation": r1.relationship_coverage,
        "covered_activity": r1.activity_coverage,
        "missing_relation": r2.relationship_coverage,
    }


def bench_pipeline() -> dict[str, object]:
    """Run one analysis if container can start; capture VLM count and timings."""
    try:
        from app.startup.orchestrator import StartupOrchestrator
        from core.contracts.pipeline import AnalysisOptions, PipelineRequest
        from language.localization.caption_translator import CaptionTranslator
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"import: {exc}"}

    image = _make_office_image(ROOT / "tmp" / "bench_pipeline.png")
    t_start = time.perf_counter()
    try:
        startup = StartupOrchestrator().run()
        orchestrator = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"startup: {exc}"}
    load_s = time.perf_counter() - t_start

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
    try:
        result = orchestrator.analyze(request)
    except Exception as exc:  # noqa: BLE001
        return {
            "skipped": True,
            "reason": f"analyze: {exc}",
            "startup_s": round(load_s, 2),
            "seconds": round(time.perf_counter() - t0, 2),
        }

    total = time.perf_counter() - t0
    stages = {m.stage.value: round(m.duration_ms, 1) for m in result.metrics.stage_metrics}
    caption = result.caption.canonical_caption_en or result.caption.narrative_full
    # Language-switch simulation: translation only, no second analyze.
    translator = CaptionTranslator()
    lang_times = {}
    for lang in ("fa", "de", "es", "zh", "en"):
        t_lang = time.perf_counter()
        translated = translator.translate(caption, lang) if lang != "en" else caption
        lang_times[lang] = {
            "ms": round((time.perf_counter() - t_lang) * 1000, 1),
            "preview": translated[:120],
        }
    return {
        "skipped": False,
        "startup_s": round(load_s, 2),
        "total_s": round(total, 2),
        "vlm_executions": result.metrics.vlm_executions,
        "caption_generation_count": result.metrics.caption_generation_count,
        "qa_count": result.metrics.qa_count,
        "caption_words": len(caption.split()),
        "caption_preview": caption[:280],
        "stages_ms": stages,
        "objects": result.metrics.objects_detected,
        "relations": result.metrics.relationships_inferred,
        "language_switch_vlm": 0,
        "language_switch_translations": translator.translation_count,
        "language_previews": lang_times,
        "sam2_note": "weights missing unless models/sam2 present",
        "qa_passed": result.qa_passed,
        "canonical_locked": bool(result.caption.canonical_caption_en),
    }


def main() -> None:
    report = {
        "german_catalog": _german_audit(),
        "enhancement": bench_enhancement(),
        "ocr": bench_ocr(),
        "metrics": bench_metrics(),
        "translation": bench_translation(),
        "pipeline": bench_pipeline(),
    }
    out = ROOT / "tmp" / "competition_readiness_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


def _german_audit() -> dict[str, object]:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    de = json.loads((ROOT / "translations" / "de.json").read_text(encoding="utf-8"))
    identical = sorted(k for k in en if en[k] == de.get(k))
    return {
        "en_keys": len(en),
        "de_keys": len(de),
        "missing": sorted(set(en) - set(de)),
        "extra": sorted(set(de) - set(en)),
        "identical_keys": identical,
        "identical_count": len(identical),
        "translated_count": len(en) - len(identical),
        "coverage_pct": round(100.0 * (len(en) - len(identical)) / max(1, len(en)), 2),
    }


if __name__ == "__main__":
    main()
