"""Real-image grounding check: caption quality + evidence-first assistant."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _ensure_ski_image() -> Path:
    out = ROOT / "tmp" / "competition_e2e_ski.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = (
        "http://images.cocodataset.org/val2017/000000322864.jpg",
        "http://images.cocodataset.org/val2017/000000581781.jpg",
        "http://images.cocodataset.org/val2017/000000143931.jpg",
    )
    for url in candidates:
        try:
            urllib.request.urlretrieve(url, out)
            if out.is_file() and out.stat().st_size > 10_000:
                return out
        except Exception:
            continue
    raise SystemExit("Could not download a test image")


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import (
        VisionAssistant,
        VisionAssistantSession,
        build_evidence_packet,
        generate_suggested_questions,
    )

    image = _ensure_ski_image()
    print(f"Image: {image}", flush=True)
    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    if hasattr(pipe._vision_language, "reset_execution_count"):
        pipe._vision_language.reset_execution_count()

    t0 = time.perf_counter()
    result = pipe.analyze(
        PipelineRequest(
            image_path=image,
            options=AnalysisOptions(
                enable_gemma=True,
                enable_enhancement=True,
                enable_sam2=True,
            ),
        )
    )
    analyze_s = time.perf_counter() - t0
    caption = result.caption.canonical_caption_en or result.caption.text
    labels = [n.label for n in result.scene_context.graph.nodes]
    vlm = result.initial_vlm_calls or result.metrics.vlm_executions
    print(f"analyze_s={analyze_s:.2f} vlm={vlm}", flush=True)
    print(f"CAPTION: {caption}", flush=True)
    print(f"LABELS: {labels}", flush=True)

    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=caption,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    suggestions = generate_suggested_questions(packet, language="en", limit=1)
    print(f"SUGGESTIONS: {suggestions}", flush=True)
    assert all("shoe" not in q.lower() for q in suggestions)

    session = VisionAssistantSession(image_key=str(image), evidence=packet)
    assistant = VisionAssistant()
    label_l = [x.lower() for x in labels]
    if any("ski" in x or "snowboard" in x for x in label_l):
        q = "What equipment is the person using?"
    else:
        # Evidence-absent-from-caption probe: ask about a secondary label not in caption.
        secondary = next((x for x in label_l if x not in caption.lower() and x not in {"person", "people"}), "")
        q = f"What can you tell about the {secondary}?" if secondary else "What objects are visible near the main subject?"
    a1 = assistant.answer(session, q, language="en")
    a_unknown = assistant.answer(session, "What is the person's exact age?", language="en")
    print(f"Q: {q}", flush=True)
    print(f"A1: {a1}", flush=True)
    print(f"UNKNOWN: {a_unknown}", flush=True)
    print(
        f"assistant_vlm={session.assistant_vlm_calls} assistant_llm={session.assistant_llm_calls}",
        flush=True,
    )

    report = {
        "image": str(image),
        "caption": caption,
        "labels": labels,
        "suggestions": suggestions,
        "question": q,
        "answer": a1,
        "unknown": a_unknown,
        "vlm": vlm,
        "assistant_vlm": session.assistant_vlm_calls,
        "assistant_llm": session.assistant_llm_calls,
        "analyze_seconds": round(analyze_s, 2),
        "no_skis_is": "skis is" not in caption.lower(),
        "no_moment_filler": "moves through the moment" not in caption.lower(),
    }
    out = ROOT / "tmp" / "ski_grounding_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    ok = bool(caption) and vlm <= 2 and session.assistant_vlm_calls == 0 and report["no_skis_is"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
