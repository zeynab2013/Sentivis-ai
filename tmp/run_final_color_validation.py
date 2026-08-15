"""FINAL COLOR FIX — real-image color entity binding validation."""

from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

CASES = [
    ("HORSE", Path("tmp/uploads/10815824_2997e03d76.jpg"), [
        "What color is the horse?",
        "What color clothing is the person wearing?",
        "What is the person doing?",
    ]),
    ("SOCCER", Path("tmp/uploads/47871819_db55ac4699.jpg"), [
        "What color is the sports ball?",
        "What color clothing is the person wearing?",
        "What are they doing?",
    ]),
    ("BICYCLE", Path("tmp/uploads/191003284_1025b0fb7d.jpg"), [
        "What color is the bicycle?",
        "What color clothing is the person wearing?",
        "What is the person doing?",
    ]),
    ("MOTORCYCLE", Path("tmp/uploads/143552829_72b6ba49d4.jpg"), [
        "What color clothing is the person wearing?",
        "What is the person doing?",
    ]),
]


def _entity_colors(result) -> list[str]:
    rows: list[str] = []
    attrs = list(getattr(result, "attributes", ()) or ())
    dets = list(getattr(result, "detections", ()) or ())
    ve = getattr(result, "verified_evidence", None)
    if ve is not None:
        for fact in getattr(ve, "facts", ()) or ():
            if getattr(fact, "predicate", "") in {"color", "dominant_color", "shirt_color", "pants_color", "shoes_color"}:
                rows.append(
                    f"  FACT {fact.entity_id} {fact.subject} {fact.predicate}={fact.object} "
                    f"[{getattr(fact, 'claim_status', '')}]"
                )
    for det in dets:
        idx = getattr(det, "object_index", None)
        label = getattr(det, "label", "?")
        color = shirt = pants = shoes = "—"
        for a in attrs:
            if getattr(a, "object_index", None) != idx:
                continue
            key = (getattr(a, "key", "") or "").lower()
            val = getattr(a, "value", "") or ""
            if key in {"color", "dominant_color"}:
                color = val
            elif key == "shirt_color":
                shirt = val
            elif key == "pants_color":
                pants = val
            elif key == "shoes_color":
                shoes = val
        rows.append(
            f"  DET {label}#{idx}: color={color} shirt={shirt} pants={pants} shoes={shoes}"
        )
    return rows


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
    print("=== FINAL COLOR VALIDATION ===\n")
    for name, path, questions in CASES:
        print(f"## {name} — {path}")
        if not path.exists():
            print("MISSING IMAGE\n")
            continue
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        caption = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        )
        print(f"CAPTION: {caption[:280]}")
        print("ENTITY COLORS:")
        for line in _entity_colors(result):
            print(line)
        packet = build_evidence_packet(
            verified_evidence=result.verified_evidence,
            canonical_caption_en=caption,
            evidence_brief=getattr(result, "evidence_brief", "") or "",
        )
        va = VisionAssistant()
        session = VisionAssistantSession(image_key=str(path), evidence=packet)
        answers: dict[str, str] = {}
        for q in questions:
            answers[q] = va.answer(session, q, language="en")
            print(f"Q: {q}")
            print(f"A: {answers[q]}")

        fails: list[str] = []
        ball = next((a for q, a in answers.items() if "sports ball" in q.lower()), "")
        bike = next(
            (a for q, a in answers.items() if "bicycle" in q.lower() and "color" in q.lower()),
            "",
        )
        doing = " ".join(a.lower() for q, a in answers.items() if "doing" in q.lower())
        cap_l = caption.lower()
        if name == "SOCCER" and any(t in ball.lower() for t in ("beige", "tan", "khaki", "olive")):
            fails.append("ball_beige_bleed")
        if name == "SOCCER" and "white" not in ball.lower() and "can't" not in ball.lower() and "cannot" not in ball.lower() and "determin" not in ball.lower():
            # Prefer white; honest refuse OK; wrong color FAIL already covered
            if any(t in ball.lower() for t in ("green", "brown", "blue", "red", "yellow")):
                fails.append("ball_wrong_color")
        if name == "BICYCLE" and any(t in bike.lower() for t in ("dark green", "olive")):
            if "can't" not in bike.lower() and "cannot" not in bike.lower() and "determin" not in bike.lower():
                fails.append("bike_green_bleed")
        if name == "BICYCLE" and "green" in bike.lower() and "can't" not in bike.lower() and "determin" not in bike.lower():
            fails.append("bike_green_bleed")
        if name == "BICYCLE" and "riding" not in doing and "bicycle" not in cap_l:
            fails.append("bike_activity_regression")
        if name == "MOTORCYCLE" and "riding" not in doing and "motorcycle" not in cap_l:
            fails.append("moto_activity_regression")
        if name == "HORSE" and "leading" not in doing and "leading" not in cap_l and "holding" not in doing:
            fails.append("horse_activity_regression")
        if name == "SOCCER" and not any(t in doing or t in cap_l for t in ("football", "soccer", "playing", "kick")):
            fails.append("soccer_activity_soft")
        print(f"COLOR_HARD_CHECKS: {'FAIL ' + ','.join(fails) if fails else 'PASS'}")
        print()


if __name__ == "__main__":
    main()
