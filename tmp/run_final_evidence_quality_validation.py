"""Final freeze validation: critical 4 + dense scenes (caption depth + metrics)."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession


def _release_between_runs(orch) -> None:
    """Free RAM between sequential analyzes so the memory guard does not trip on CPU demos."""
    try:
        mm = getattr(orch, "_memory", None)
        if mm is not None and hasattr(mm, "clear_gpu_cache"):
            mm.clear_gpu_cache()
    except Exception:
        pass
    gc.collect()


CRITICAL = [
    (
        "HORSE",
        Path("tmp/uploads/10815824_2997e03d76.jpg"),
        ["How many people are visible?", "What is the person doing?", "What color is the horse?"],
        ("leading", "holding"),
    ),
    (
        "SOCCER",
        Path("tmp/uploads/47871819_db55ac4699.jpg"),
        ["How many people are visible?", "What are they doing?", "What readable text appears in the scene?"],
        ("football", "soccer", "playing"),
    ),
    (
        "MOTORCYCLE",
        Path("tmp/uploads/143552829_72b6ba49d4.jpg"),
        ["How many people are visible?", "What is the person doing?"],
        ("riding", "motorcycle"),
    ),
    (
        "BICYCLE",
        Path("tmp/uploads/191003284_1025b0fb7d.jpg"),
        ["How many people are visible?", "What is the person doing?"],
        ("riding", "bicycle"),
    ),
]

DENSE = [
    ("DENSE_TENNIS", Path("tmp/uploads/random_385406.jpg")),
    ("KITCHEN", Path("tmp/coco_kitchen.jpg")),
]


def main() -> None:
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    rows = []
    fails = 0
    for name, path, questions, need in CRITICAL:
        if not path.exists():
            rows.append({"name": name, "status": "MISSING"})
            fails += 1
            continue
        print(f"=== {name} ===", flush=True)
        _release_between_runs(orch)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        cap = getattr(result.caption, "canonical_caption_en", None) or result.caption.text or ""
        cap_l = cap.lower()
        ve = result.verified_evidence
        packet = build_evidence_packet(
            verified_evidence=ve,
            canonical_caption_en=cap,
            evidence_brief=getattr(result, "evidence_brief", "") or "",
        )
        va = VisionAssistant()
        session = VisionAssistantSession(image_key=str(path), evidence=packet)
        answers = {q: va.answer(session, q, language="en") for q in questions}
        act_ok = any(t in cap_l for t in need) or any(
            any(t in a.lower() for t in need) for a in answers.values()
        )
        status = "PASS" if act_ok else "FAIL"
        if status == "FAIL":
            fails += 1
        qr = result.quality_report
        rows.append(
            {
                "name": name,
                "status": status,
                "words": len(cap.split()),
                "caption": cap,
                "answers": answers,
                "object_coverage": None if qr is None else qr.object_coverage,
                "relationship_coverage": None if qr is None else qr.relationship_coverage,
                "activity_coverage": None if qr is None else qr.activity_coverage,
                "hallucination_risk": None if qr is None else qr.hallucination_risk,
                "confirmed_activities": [
                    a.activity
                    for a in (ve.activities if ve else [])
                    if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
                ],
            }
        )
        print(f"{status} words={len(cap.split())}", flush=True)
        print(cap, flush=True)

    for name, path in DENSE:
        if not path.exists():
            rows.append({"name": name, "status": "MISSING"})
            continue
        print(f"=== {name} ===", flush=True)
        _release_between_runs(orch)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        cap = getattr(result.caption, "canonical_caption_en", None) or result.caption.text or ""
        ve = result.verified_evidence
        labels = sorted(
            {(e.label or "").lower() for e in (ve.entities if ve else []) if (e.label or "").strip()}
        )
        qr = result.quality_report
        rows.append(
            {
                "name": name,
                "status": "OK",
                "words": len(cap.split()),
                "caption": cap,
                "verified_labels": labels,
                "setting": getattr(getattr(ve, "scene", None), "setting", "") if ve else "",
                "object_coverage": None if qr is None else qr.object_coverage,
                "relationship_coverage": None if qr is None else qr.relationship_coverage,
            }
        )
        print(f"words={len(cap.split())} labels={labels}", flush=True)
        print(cap, flush=True)

    out = Path("tmp/FINAL_EVIDENCE_QUALITY_VALIDATION.json")
    out.write_text(json.dumps({"fails": fails, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {out} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
