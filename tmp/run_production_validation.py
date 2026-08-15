"""Production validation pass — real diverse images only. No product code changes."""

from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

# Real photographic images (assets/samples/* are abstract demo placeholders)
CASES = [
    ("1_kitchen", Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")),
    ("2_animal", Path("tmp/uploads/random_850976.jpg")),  # brown bear
    ("3_vehicle", Path("tmp/competition_e2e_street.jpg")),  # campaign bus
    ("4_sports", Path("tmp/uploads/random_240850.jpg")),  # baseball players
    ("5_landscape", Path("tmp/uploads/95728660_d47de66544.jpg")),  # mountain trail
    ("6_multiple_people", Path("tmp/uploads/random_385406.jpg")),  # tennis group ~12
]

QUESTIONS = [
    "How many people are visible?",
    "What is the person doing?",
    "What objects are visible?",
    "What colors are visible?",
    "Where is the main object/person?",
]

_PERSON = {"person", "man", "woman", "child", "people"}


def main() -> None:
    out = Path("tmp/production_validation_report.txt")
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=True,
        competition_mode=False,
        enable_enhancement=True,
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
        log(f"EXISTS: {image.exists()}")
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
        log("--- VERIFIED SCENE ---")
        if ve is None:
            log("(no verified evidence)")
            continue
        log(
            f"scene_type={ve.scene.scene_type!r} setting={ve.scene.setting!r} "
            f"indoor_outdoor={ve.scene.indoor_outdoor!r} conf={ve.scene.confidence:.2f}"
        )
        log("--- VERIFIED ACTIVITIES (all) ---")
        acts = list(ve.activities)
        if not acts:
            log("(none)")
        for a in acts:
            log(
                f"  activity={a.activity!r} conf={a.confidence:.2f} "
                f"qa_safe={a.qa_safe} narrative_safe={a.narrative_safe} "
                f"status={a.status.value} support={a.supporting_relations} "
                f"entities={a.entity_ids}"
            )
        log("--- QA-SAFE ACTIVITIES ONLY ---")
        qa_acts = [a for a in acts if a.qa_safe]
        if not qa_acts:
            log("(none)")
        for a in qa_acts:
            log(f"  {a.activity!r} conf={a.confidence:.2f}")
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
        log("--- QA ANSWERS ---")
        answers: dict[str, str] = {}
        for q in QUESTIONS:
            session = VisionAssistantSession(image_key=name, evidence=packet)
            try:
                ans = va.answer(session, q)
            except Exception as exc:  # noqa: BLE001
                ans = f"ERROR: {exc}"
            answers[q] = ans
            log(f"Q: {q}")
            log(f"A: {ans}")
            log("---")
        people = [e.entity_id for e in ve.entities if e.label.lower() in _PERSON]
        labels = sorted({e.label for e in ve.entities if e.narrative_safe})
        cap_l = cap.lower()
        halluc = [
            a.activity
            for a in qa_acts
            if a.activity.lower() not in cap_l
            and a.activity.lower() not in {"standing", "sitting", "walking"}
        ]
        log("--- QUICK FLAGS ---")
        log(f"people_entities={people}")
        log(f"object_labels={labels[:30]}")
        log(f"qa_safe_activities_absent_from_caption={halluc}")
        joined_s = " ".join(suggestions).lower()
        if "what is the person doing" in joined_s and not qa_acts:
            log("FLAG: suggested activity question but no qa_safe activity")
        doing = answers.get("What is the person doing?", "").lower()
        if any(
            bad in doing
            for bad in ("office work", "classroom learning", "meeting", "teaching")
        ):
            log(f"FLAG: possible hallucinated activity answer: {doing}")
        if ve.scene.scene_type in {"restaurant", "office", "classroom"} and any(
            lab in labels for lab in ("refrigerator", "oven", "sink")
        ):
            log(
                f"FLAG: scene={ve.scene.scene_type} despite kitchen appliances "
                f"in entities"
            )
        log("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("REPORT_WRITTEN", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
