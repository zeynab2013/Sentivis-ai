"""Quick competition readiness probe on one real image."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

IMAGE = ROOT / "tmp" / "uploads" / "random_268119.jpg"
OUT = ROOT / "tmp" / "competition_ready_probe.json"


def main() -> int:
    import psutil

    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import build_evidence_packet, generate_suggested_questions
    from language.tts import synthesize_display_text

    if not IMAGE.is_file():
        raise SystemExit(f"missing {IMAGE}")

    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    result = pipe.analyze(
        PipelineRequest(
            image_path=IMAGE,
            options=AnalysisOptions(enable_gemma=True, enable_enhancement=True, enable_sam2=False),
        )
    )
    ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=result.caption.canonical_caption_en or result.caption.text,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    suggestions = generate_suggested_questions(packet, language="en", limit=3)
    colors = [
        f"{a.object_index}:{a.name}={a.value}"
        for a in result.scene_context.attributes.attributes
        if "color" in a.name.lower() and a.value.lower() not in {"unknown", ""}
    ][:12]
    relations = [
        f"{r.subject_index}-{r.relation_type}-{r.object_index}"
        for r in result.scene_context.graph.relations[:12]
    ]
    caption = result.caption.canonical_caption_en or result.caption.text
    tts = synthesize_display_text(caption[:180], "en")
    iq = result.image_quality
    payload = {
        "image": IMAGE.name,
        "caption": caption,
        "robotic_failure": "a person talking to a person" in caption.lower(),
        "suggestions": suggestions,
        "colors": colors,
        "relations": relations,
        "enhancement": {
            "level": getattr(iq, "quality_level", None),
            "applied": getattr(iq, "enhancement_applied", None),
            "ops": list(getattr(iq, "enhancement_operations", ()) or ()),
            "rejected": getattr(iq, "enhancement_rejected", None),
            "reason": getattr(iq, "rejection_reason", None),
        },
        "tts_bytes": None if tts is None else len(tts),
        "ram_mb_before": round(ram_before, 1),
        "ram_mb_after": round(ram_after, 1),
        "objects": [n.label for n in result.scene_context.graph.nodes[:15]],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:2500])
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
