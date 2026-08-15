"""Production reliability validation — diverse real images including bicycle/handbag failure case."""

from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from ui.formatters.result_formatters import format_activities, format_relationships, format_scene_summary

CASES = [
    ("bike_handbag_regression", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("kitchen", Path("tmp/coco_kitchen.jpg")),
    ("vehicle_street", Path("tmp/competition_e2e_street.jpg")),
    ("sports_baseball", Path("tmp/coco_baseball.jpg")),
    ("ski", Path("tmp/competition_e2e_ski.jpg")),
    ("multi_people_tennis", Path("tmp/uploads/random_385406.jpg")),
    ("animal_bear", Path("tmp/uploads/random_850976.jpg")),
    ("landscape", Path("tmp/uploads/95728660_d47de66544.jpg")),
]

QUESTIONS = [
    "How many people are visible?",
    "What is the person doing?",
    "What objects are visible?",
]


def main() -> None:
    out = Path("tmp/production_reliability_audit.txt")
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=True,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    va = VisionAssistant()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    for name, image in CASES:
        log("=" * 80)
        log(f"CASE: {name}")
        log(f"IMAGE: {image}")
        if not image.exists():
            log("SKIP missing image")
            continue
        try:
            result = orch.analyze(PipelineRequest(image_path=image, options=opts))
        except Exception as exc:  # noqa: BLE001
            log(f"PIPELINE ERROR: {exc}")
            continue
        ve = result.verified_evidence
        cap = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        ).strip()
        log("--- FINAL CAPTION ---")
        log(cap)
        if ve is None:
            log("(no verified evidence)")
            continue
        log("--- HUMAN SCENE SUMMARY ---")
        log(ve.compose_human_scene_summary())
        log("--- VERIFIED RELATIONS ---")
        for r in ve.qa_relations()[:12]:
            log(
                f"  {r.subject_id} {r.relation_type} {r.object_id} "
                f"conf={r.confidence:.2f} narr={r.narrative_safe} qa={r.qa_safe}"
            )
        log("--- ACTIVITIES BY TIER ---")
        for a in ve.activities:
            log(
                f"  [{a.evidence_level.value}] {a.activity!r} entities={a.entity_ids} "
                f"qa={a.qa_safe} narr={a.narrative_safe}"
            )
        log("--- REPORT FORMATTERS ---")
        log(format_scene_summary(result))
        log(format_relationships(result))
        log(format_activities(result))
        packet = build_evidence_packet(
            result.scene_context,
            canonical_caption_en=cap,
            evidence_brief=result.evidence_brief,
            ocr_snippets=result.ocr_snippets,
            verified_evidence=ve,
        )
        suggestions = generate_suggested_questions(packet)
        log("--- SUGGESTED QUESTIONS ---")
        for q in suggestions:
            log(f"  - {q}")
        log("--- QA ---")
        for q in QUESTIONS:
            session = VisionAssistantSession(image_key=name, evidence=packet)
            try:
                ans = va.answer(session, q)
            except Exception as exc:  # noqa: BLE001
                ans = f"ERROR: {exc}"
            log(f"Q: {q}")
            log(f"A: {ans}")

        # Hard checks for the known failure case.
        if name == "bike_handbag_regression":
            lower = cap.lower()
            act_names = " ".join(a.activity.lower() for a in ve.activities if a.qa_safe)
            log("--- REGRESSION ASSERTIONS ---")
            checks = {
                "no_gender_man_woman": (" man " not in f" {lower} " and " woman " not in f" {lower} "),
                "no_shoulder_invention": "shoulder" not in lower,
                "no_shopping_activity": "shopping" not in act_names,
                "no_driving_activity": "driving" not in act_names,
                "no_bicycle_inside_person": not any(
                    r.relation_type == "inside"
                    and "bicycle" in (r.subject_id + r.object_id)
                    and "person" in (r.subject_id + r.object_id)
                    for r in ve.relations
                    if r.qa_safe or r.narrative_safe
                ),
                "has_riding_or_bicycle_mention": (
                    "riding" in lower
                    or "bicycle" in lower
                    or any(
                        a.evidence_level == ActivityEvidenceLevel.CONFIRMED
                        and "bicycle" in a.activity.lower()
                        for a in ve.activities
                    )
                    or any(r.relation_type == "riding" for r in ve.narrative_relations())
                ),
            }
            for key, ok in checks.items():
                log(f"  {key}: {'PASS' if ok else 'FAIL'}")

    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
