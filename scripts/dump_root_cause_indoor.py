"""Root-cause dump for one real failing indoor image."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

IMAGE = ROOT / "tmp" / "uploads" / "random_268119.jpg"
OUT = ROOT / "tmp" / "root_cause_indoor_dump.json"


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import (
        build_evidence_packet,
        generate_suggested_questions,
    )
    from language.assistant.suggested_questions import _build_candidates, _simulate_answerable
    from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
    from language.semantic.natural_caption_service import NaturalCaptionService

    if not IMAGE.is_file():
        raise SystemExit(f"Missing image: {IMAGE}")

    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    if hasattr(pipe._vision_language, "reset_execution_count"):
        pipe._vision_language.reset_execution_count()

    result = pipe.analyze(
        PipelineRequest(
            image_path=IMAGE,
            options=AnalysisOptions(
                enable_gemma=True,
                enable_enhancement=True,
                enable_sam2=True,
            ),
        )
    )

    ctx = result.scene_context
    nodes = [
        {
            "id": n.object_id,
            "index": n.index,
            "label": n.label,
            "area": n.area,
            "position": n.position,
        }
        for n in ctx.graph.nodes
    ]
    relations = [
        {
            "subject": r.subject_index,
            "object": r.object_index,
            "predicate": r.predicate,
            "confidence": r.confidence,
        }
        for r in ctx.graph.relations
    ]
    attrs = [
        {"object_index": a.object_index, "name": a.name, "value": a.value}
        for a in ctx.attributes.attributes
    ]
    activities = [
        {"activity": a.activity, "confidence": a.confidence, "objects": list(a.object_indices)}
        for a in ctx.activities.activities
    ]

    # Reconstruct story path used by caption service.
    understanding = getattr(result, "understanding", None)
    story_dump = {}
    sentences = {}
    if understanding is not None:
        svc = NaturalCaptionService(pipe._vision_language)  # type: ignore[arg-type]
        scene = svc._build_semantic_scene(understanding)
        story = svc._story_facts(understanding, scene=scene)
        brief = svc._build_understanding_brief(story, scene, understanding)
        story_dump = {
            "scene_type": story.scene_type,
            "people": list(story.people),
            "main": story.main,
            "main_label": story.main_label,
            "action": story.action,
            "interaction": story.primary_interaction,
            "objects": list(story.objects),
            "background": list(story.background_objects),
            "place": story.place,
            "relations": list(story.relations),
            "person_should_lead": svc._person_should_lead(understanding, list(story.people) or svc._people(understanding)),
            "ranked_subjects": list(understanding.ranked_subjects),
            "evidence_brief": understanding.evidence_brief,
        }
        sentences = {
            "main_event": svc._sentence_main_event(story, brief),
            "interaction": svc._sentence_interaction(story, brief),
            "people_animals": svc._sentence_people_animals(story, brief),
            "objects": svc._sentence_objects(story, brief, already_narrated=""),
            "background": svc._sentence_background(story, brief, scene),
            "environment": svc._sentence_environment(story, brief, scene),
            "atmosphere": svc._sentence_atmosphere(story, brief),
            "closing": svc._sentence_closing(story, brief),
            "composed": svc._compose_scene_narrative(story, scene=scene, brief=brief),
        }

    packet = build_evidence_packet(
        ctx,
        canonical_caption_en=result.caption.canonical_caption_en or result.caption.text,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    caption_l = (packet.canonical_caption_en or "").lower()
    retriever = VisualEvidenceRetriever()
    candidates = _build_candidates(packet, caption_l)
    candidate_trace = []
    for score, question, cues in candidates:
        answerable = _simulate_answerable(packet, question, retriever)
        from language.assistant.suggested_questions import _is_caption_duplicate, _MIN_SCORE

        rejected = None
        if score < _MIN_SCORE:
            rejected = f"score<{_MIN_SCORE}"
        elif _is_caption_duplicate(question, packet.canonical_caption_en or "", cues):
            rejected = "caption_duplicate"
        elif not answerable:
            rejected = "not_answerable"
        candidate_trace.append(
            {
                "score": score,
                "question": question,
                "cues": list(cues),
                "answerable": answerable,
                "rejected": rejected,
            }
        )
    suggestions = generate_suggested_questions(packet, language="en", limit=1)

    dump = {
        "image": str(IMAGE),
        "yolo_nodes": nodes,
        "relations": relations,
        "attributes_sample": attrs[:40],
        "activities": activities,
        "environment": {
            "setting": ctx.environment.setting,
            "indoor_outdoor": ctx.environment.indoor_outdoor,
            "scene_type": ctx.environment.scene_type,
        },
        "ocr": list(result.ocr_snippets or ()),
        "sam2": "unavailable/disabled unless weights present",
        "evidence_brief": result.evidence_brief,
        "story": story_dump,
        "sentence_builders": sentences,
        "canonical_caption_en": result.caption.canonical_caption_en or result.caption.text,
        "suggested_candidates": candidate_trace,
        "suggested_final": suggestions,
        "vlm_calls": result.initial_vlm_calls or result.metrics.vlm_executions,
        "person_in_nodes": any(n["label"] == "person" for n in nodes),
        "person_in_caption": "person" in (result.caption.canonical_caption_en or result.caption.text or "").lower()
        or "man" in (result.caption.canonical_caption_en or result.caption.text or "").lower()
        or "woman" in (result.caption.canonical_caption_en or result.caption.text or "").lower(),
    }
    OUT.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(json.dumps(dump, indent=2)[:6000])
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
