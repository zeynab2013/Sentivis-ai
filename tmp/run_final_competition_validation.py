"""Final competition production validation — diverse real images."""

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
    ("indoor_kitchen", Path("tmp/coco_kitchen.jpg")),
    ("outdoor_bike_handbag", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("vehicle_bus", Path("tmp/competition_e2e_street.jpg")),
    ("sports_baseball", Path("tmp/coco_baseball.jpg")),
    ("sports_ski", Path("tmp/competition_e2e_ski.jpg")),
    ("multi_people", Path("tmp/uploads/random_385406.jpg")),
    ("animal_bear", Path("tmp/uploads/random_850976.jpg")),
    ("landscape", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("low_quality_blur", Path("tmp/enhance_blur.jpg")),
    ("enhanced_preview", Path("tmp/pipeline_enhanced_500x333.png")),
]

QUESTIONS = [
    "How many people are visible?",
    "What is the person doing?",
    "What objects are visible?",
]

_BANNED_IF_WEAK = (
    "shopping",
    "driving",
    "playing tennis",
    "office work",
    "restaurant dining",
    "crossing a street",
    "preparing a campfire",
    "walking a dog",
)


def main() -> None:
    out = Path("tmp/final_competition_validation.txt")
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    va = VisionAssistant()
    lines: list[str] = []
    failures: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    for name, image in CASES:
        log("=" * 80)
        log(f"CASE: {name}")
        log(f"IMAGE: {image}")
        if not image.exists():
            log("SKIP missing")
            continue
        try:
            result = orch.analyze(PipelineRequest(image_path=image, options=opts))
        except Exception as exc:  # noqa: BLE001
            log(f"PIPELINE ERROR: {exc}")
            failures.append(f"{name}: pipeline error")
            continue
        ve = result.verified_evidence
        cap = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        ).strip()
        log("--- CAPTION ---")
        log(cap)
        if ve is None:
            log("NO VERIFIED EVIDENCE")
            failures.append(f"{name}: missing verified evidence")
            continue
        log("--- SUMMARY ---")
        log(ve.compose_human_scene_summary())
        log("--- RELATIONS ---")
        for r in ve.narrative_relations()[:8]:
            log(f"  {r.subject_id} {r.relation_type} {r.object_id}")
        log("--- ACTIVITIES ---")
        for a in ve.activities:
            log(
                f"  [{a.evidence_level.value}] {a.activity!r} "
                f"qa={a.qa_safe} narr={a.narrative_safe}"
            )
        log("--- REPORT ---")
        log(format_scene_summary(result))
        log(format_relationships(result))
        log(format_activities(result))
        packet = build_evidence_packet(
            result.scene_context,
            canonical_caption_en=cap,
            verified_evidence=ve,
        )
        suggestions = generate_suggested_questions(packet)
        log("--- SUGGESTIONS ---")
        for q in suggestions:
            log(f"  - {q}")
        log("--- QA ---")
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

        # Consistency checks
        narr_acts = {
            a.activity.lower()
            for a in ve.activities
            if a.narrative_safe
            and a.evidence_level == ActivityEvidenceLevel.CONFIRMED
        }
        qa_acts = {
            a.activity.lower()
            for a in ve.activities
            if a.qa_safe
            and a.evidence_level
            in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        }
        cap_l = cap.lower()
        # Caption must not claim banned weak activities unless verified.
        for banned in _BANNED_IF_WEAK:
            if banned in cap_l and not any(banned in a for a in qa_acts | narr_acts):
                failures.append(f"{name}: caption claims unverified '{banned}'")
                log(f"FAIL caption unverified activity: {banned}")
        # Narrative activities must be QA-answerable.
        for act in narr_acts:
            token = next((t for t in act.split() if len(t) > 4), act)
            if token and token not in " ".join(qa_acts) and token not in answers.get(
                "What is the person doing?", ""
            ).lower():
                # Soft warn — may be multi-person scenes with different focus.
                log(f"WARN narrative activity maybe missing in QA: {act}")
        # Suggestions must not ask banned activities unless verified.
        sug = " ".join(suggestions).lower()
        for banned in _BANNED_IF_WEAK:
            if banned in sug and not any(banned in a for a in qa_acts):
                failures.append(f"{name}: suggestion asks unverified '{banned}'")
                log(f"FAIL suggestion unverified: {banned}")
        # Scene overclaims
        st = (ve.scene.scene_type or "").lower()
        se = (ve.scene.setting or "").lower()
        labels = {e.label.lower() for e in ve.entities if e.narrative_safe}
        if "farm" in st or "farm" in se:
            livestock = labels & {"cow", "horse", "sheep", "goat"}
            if len(livestock) < 2:
                failures.append(f"{name}: farm from weak livestock")
                log("FAIL farm overclaim")
        if "highway" in st or "highway" in se:
            if not labels & {"road", "traffic light", "stop sign"}:
                failures.append(f"{name}: highway without road cues")
                log("FAIL highway overclaim")
        if "tennis court" in st or "tennis court" in se:
            if not any("playing tennis" in a for a in qa_acts):
                failures.append(f"{name}: tennis court without verified play")
                log("FAIL tennis court overclaim")
        if "office" in st or "office" in se:
            if len(labels & {"laptop", "keyboard", "mouse"}) < 2:
                failures.append(f"{name}: office from single device")
                log("FAIL office overclaim")

    log("=" * 80)
    log(f"FAILURES: {len(failures)}")
    for f in failures:
        log(f"  - {f}")
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
