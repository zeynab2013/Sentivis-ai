"""Probe the farm image that previously produced broken grammar captions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

IMAGE = ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg"
OUT = ROOT / "tmp" / "farm_stage2_probe.json"


def main() -> int:
    import psutil

    from app.startup.orchestrator import StartupOrchestrator
    from analysis.relationships.relation_metrics import meaningful_relations
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import build_evidence_packet, generate_suggested_questions
    from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
    from ui.formatters.result_formatters import format_detected_objects, format_relationships

    if not IMAGE.is_file():
        raise SystemExit(f"missing {IMAGE}")

    ram0 = psutil.Process().memory_info().rss / (1024 * 1024)
    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    result = pipe.analyze(
        PipelineRequest(
            image_path=IMAGE,
            options=AnalysisOptions(enable_gemma=True, enable_enhancement=True, enable_sam2=False),
        )
    )
    ram1 = psutil.Process().memory_info().rss / (1024 * 1024)
    caption = result.caption.canonical_caption_en or result.caption.text
    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=caption,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    suggestions = generate_suggested_questions(packet, language="en", limit=5)
    grammar = CaptionQualityEvaluator()._grammar_score(caption)
    iq = result.image_quality
    colors = [
        f"{a.object_index}:{a.name}={a.value}"
        for a in result.scene_context.attributes.attributes
        if "color" in a.name.lower() and str(a.value).lower() not in {"unknown", ""}
    ][:16]
    payload = {
        "image": IMAGE.name,
        "caption": caption,
        "grammar_score": round(grammar, 3),
        "has_with_one_is": "with one is" in caption.lower(),
        "robotic_failure": "a person talking to a person" in caption.lower(),
        "suggestions": suggestions,
        "colors": colors,
        "relationships_ui": format_relationships(result),
        "relationships_metric": result.metrics.relationships_inferred,
        "meaningful_relation_count": len(meaningful_relations(result.scene_context.graph)),
        "objects_metric": result.metrics.objects_detected,
        "objects_ui": format_detected_objects(result),
        "enhancement": {
            "level": getattr(iq, "quality_level", None),
            "applied": getattr(iq, "enhancement_applied", None),
            "true_sr": getattr(iq, "super_resolution_used", None),
            "before": getattr(iq, "before_quality", None),
            "after": getattr(iq, "after_quality", None),
            "improvement": getattr(iq, "improvement_percent", None),
            "ops": list(getattr(iq, "enhancement_operations", ()) or ()),
            "reason": getattr(iq, "rejection_reason", None),
        },
        "ram_mb": {"before": round(ram0, 1), "after": round(ram1, 1)},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:3000])
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
