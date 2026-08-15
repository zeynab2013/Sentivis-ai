"""Final competition readiness: real pipeline + assistant + model audit.

Runs against a real image (downloads a public COCO sample if needed).
Does not start Streamlit (see separate browser E2E).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _ensure_sample_image() -> Path:
    out = ROOT / "tmp" / "competition_e2e_street.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Force refresh: prior runs accidentally used a kitchen COCO id.
    # Prefer a true street / transit scene (bus + person).
    candidates = (
        "http://images.cocodataset.org/val2017/000000143931.jpg",  # bus + person
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # kitchen multi-object
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats / colors
    )
    for url in candidates:
        try:
            urllib.request.urlretrieve(url, out)
            if out.is_file() and out.stat().st_size > 10_000:
                return out
        except Exception:
            continue
    import numpy as np
    from PIL import Image

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :] = (90, 120, 160)
    arr[200:400, 100:250] = (40, 40, 40)
    arr[280:380, 300:520] = (120, 40, 70)
    arr[0:120, :] = (40, 100, 50)
    Image.fromarray(arr).save(out, quality=85)
    return out


def main() -> int:
    from app.console_bootstrap import prepare_windows_console, print_runtime_diagnostics
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import (
        VisionAssistant,
        VisionAssistantSession,
        build_evidence_packet,
        generate_suggested_questions,
    )
    from language.localization.caption_translator import CaptionTranslator
    from language.vlm.managed_vision_model import ManagedVisionModel
    from vision.enhancement.luminance import mean_luminance

    prepare_windows_console()
    print("=== Sentivis competition readiness E2E ===", flush=True)
    print_runtime_diagnostics(root=ROOT)

    started = time.perf_counter()
    result_startup = StartupOrchestrator().run()
    print(f"Startup ready={result_startup.report.ready} warnings={len(result_startup.report.warnings)}", flush=True)

    image = _ensure_sample_image()
    print(f"Test image: {image}", flush=True)

    controller = result_startup.context.main_controller
    # Reset VLM counter if available
    orch = controller.pipeline._orchestrator  # noqa: SLF001
    if hasattr(orch._vision_language, "reset_execution_count"):
        orch._vision_language.reset_execution_count()

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
    pipeline_result = orch.analyze(request)
    analyze_s = time.perf_counter() - t0

    caption = pipeline_result.caption.canonical_caption_en or pipeline_result.caption.text
    words = len(caption.split())
    vlm = pipeline_result.initial_vlm_calls or pipeline_result.metrics.vlm_executions
    iq = pipeline_result.image_quality

    # Enhancement luminance if enhanced preview exists
    enh_note = "n/a"
    if iq is not None:
        enh_note = (
            f"level={iq.quality_level} applied={iq.enhancement_applied} "
            f"rejected={getattr(iq, 'enhancement_rejected', False)} "
            f"sr={iq.super_resolution_used} ops={list(iq.enhancement_operations)}"
        )

    packet = build_evidence_packet(
        pipeline_result.scene_context,
        canonical_caption_en=caption,
        evidence_brief=pipeline_result.evidence_brief,
        ocr_snippets=pipeline_result.ocr_snippets,
        verified_evidence=pipeline_result.verified_evidence,
    )
    suggestions = generate_suggested_questions(packet, language="en", limit=1)
    session = VisionAssistantSession(image_key=str(image), evidence=packet)
    assistant = VisionAssistant()
    a1 = assistant.answer(session, "What objects are visible near the main subject?", language="en")
    a2 = assistant.answer(session, "Where is it relative to the person?", language="en")
    a3 = assistant.answer(session, "What is the person's exact age?", language="en")

    translator = CaptionTranslator()
    translations = {}
    for lang in ("de", "es", "fa", "zh"):
        translations[lang] = translator.translate(caption, lang)[:180]

    report = {
        "startup_ready": result_startup.report.ready,
        "device_cuda": result_startup.environment.cuda_available,
        "analyze_seconds": round(analyze_s, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "image": str(image),
        "caption": caption,
        "caption_words": words,
        "objects": pipeline_result.metrics.objects_detected,
        "relations": pipeline_result.metrics.relationships_inferred,
        "activities": pipeline_result.metrics.activities_inferred,
        "vlm_executions": vlm,
        "qa_passed": pipeline_result.qa_passed,
        "enhancement": enh_note,
        "ocr_snippets": list(pipeline_result.ocr_snippets),
        "suggested_questions": suggestions,
        "assistant_q1": a1,
        "assistant_followup": a2,
        "assistant_unknown": a3,
        "assistant_vlm_calls": session.assistant_vlm_calls,
        "assistant_llm_calls": session.assistant_llm_calls,
        "translations_preview": translations,
        "relationship_coverage": pipeline_result.quality_report.relationship_coverage,
        "activity_coverage": pipeline_result.quality_report.activity_coverage,
        "sam2_note": "disabled unless weights present (checked at startup)",
    }
    out = ROOT / "tmp" / "competition_final_e2e_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {out}", flush=True)

    ok = (
        bool(caption)
        and words >= 12
        and vlm <= 2
        and session.assistant_vlm_calls == 0
        and session.assistant_llm_calls >= 3
        and pipeline_result.qa_passed
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
