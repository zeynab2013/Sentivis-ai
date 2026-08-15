"""FINAL COMPETITION FREEZE — validation only (no pipeline mutations)."""

from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

CASES = [
    (
        "HORSE",
        Path("tmp/uploads/10815824_2997e03d76.jpg"),
        [
            "How many people are visible?",
            "What is the person doing?",
            "What color clothing is the person wearing?",
            "What color is the horse?",
            "What other animals are visible?",
            "Is there fire or smoke visible in the scene?",
        ],
        {
            "need_activity": ("leading", "holding"),
            "forbid_caption": (
                "observed activity:",
                "the location is outdoor",
                "khaki-colored person",
            ),
            "forbid_qa_append": True,
        },
    ),
    (
        "SOCCER",
        Path("tmp/uploads/47871819_db55ac4699.jpg"),
        [
            "How many people are visible?",
            "What are they doing?",
            "What color is the sports ball?",
            "What color clothing is the person wearing?",
            "What readable text appears in the scene?",
        ],
        {
            "need_activity": ("football", "soccer", "playing"),
            "forbid_caption": ("observed activity:", "the location is outdoor"),
            "forbid_qa_append": True,
            "need_ocr": "21",
        },
    ),
    (
        "MOTORCYCLE",
        Path("tmp/uploads/143552829_72b6ba49d4.jpg"),
        [
            "How many people are visible?",
            "What are they doing?",
            "What is the person doing?",
            "What color clothing is the person wearing?",
        ],
        {
            "need_activity": ("riding", "motorcycle"),
            "forbid_caption": (
                "observed activity:",
                "standing beside",
            ),
            "forbid_qa_append": True,
            "prefer_shirt": ("red",),
        },
    ),
    (
        "BICYCLE",
        Path("tmp/uploads/191003284_1025b0fb7d.jpg"),
        [
            "How many people are visible?",
            "What is the person doing?",
            "What color is the bicycle?",
            "What color clothing is the person wearing?",
        ],
        {
            "need_activity": ("riding", "bicycle"),
            "forbid_caption": (
                "observed activity:",
                "the location is outdoor",
                "person, and bicycle",
            ),
            "forbid_qa_append": True,
            "forbid_bike_color_guess": ("dark green", "olive", "beige"),
        },
    ),
]


def _checks(name: str, caption: str, answers: dict[str, str], rules: dict) -> dict[str, str]:
    cap_l = caption.lower()
    out: dict[str, str] = {}

    bad_meta = [b for b in rules.get("forbid_caption", ()) if b in cap_l]
    out["caption_naturalness"] = "FAIL" if bad_meta else "PASS"
    if bad_meta:
        out["caption_naturalness_detail"] = ",".join(bad_meta)

    need = rules.get("need_activity", ())
    act_ok = any(tok in cap_l for tok in need)
    doing = " ".join(
        a.lower()
        for q, a in answers.items()
        if "doing" in q.lower() or "are they" in q.lower()
    )
    act_qa = any(tok in doing for tok in need) if doing else False
    out["activity"] = "PASS" if act_ok and act_qa else "FAIL"

    people_a = next((a for q, a in answers.items() if "how many people" in q.lower()), "")
    out["people_count"] = (
        "PASS"
        if people_a
        and ("people" in people_a.lower() or "person" in people_a.lower())
        and "can't" not in people_a.lower()
        else "FAIL"
    )

    if rules.get("forbid_qa_append"):
        appended = False
        for a in answers.values():
            if "\n\n" in a.strip() and len(a) > 160:
                appended = True
            if (
                caption
                and len(caption) > 40
                and caption[:40].lower() in a.lower()
                and not a.lower().startswith(caption[:40].lower())
            ):
                appended = True
        out["qa_no_caption_append"] = "FAIL" if appended else "PASS"

    color_fail = False
    for q, a in answers.items():
        al = a.lower()
        if name == "HORSE" and "clothing" in q.lower():
            if "khaki" in al and "light blue" not in al and "red" not in al and "blue" not in al:
                color_fail = True
        if "sports ball" in q.lower() and any(c in al for c in ("beige", "tan", "olive", "khaki")):
            color_fail = True
        if "bicycle" in q.lower() and "color" in q.lower():
            if any(c in al for c in rules.get("forbid_bike_color_guess", ())):
                color_fail = True
        if name == "MOTORCYCLE" and "clothing" in q.lower():
            prefs = rules.get("prefer_shirt", ())
            if prefs and not any(p in al for p in prefs) and "can't" not in al:
                color_fail = True
    out["colors"] = "FAIL" if color_fail else "PASS"

    if rules.get("need_ocr"):
        ocr_a = next(
            (a for q, a in answers.items() if "readable" in q.lower() or "text" in q.lower()),
            "",
        )
        out["ocr"] = "PASS" if rules["need_ocr"] in ocr_a else "FAIL"
    else:
        out["ocr"] = "N/A"

    hall = any(
        tok in cap_l
        for tok in ("observed activity:", "confirmed:", "entity:", "bbox:", "confidence:")
    )
    out["no_hallucination"] = "FAIL" if hall or bad_meta else "PASS"
    out["relationships"] = "PASS" if act_ok else "FAIL"
    return out


def main() -> None:
    import gc

    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    lines: list[str] = ["# FINAL COMPETITION FREEZE VALIDATION", ""]
    overall_fail = 0
    for name, path, questions, rules in CASES:
        if not path.exists():
            lines.append(f"## {name}")
            lines.append(f"MISSING IMAGE: `{path}`")
            lines.append("")
            overall_fail += 1
            continue
        print(f"=== {name} {path} ===", flush=True)
        gc.collect()
        try:
            result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        except Exception as exc:  # noqa: BLE001
            print(f"ANALYZE_ERROR: {exc}", flush=True)
            lines.append(f"## IMAGE: {name}")
            lines.append(f"- Path: `{path}`")
            lines.append(f"- ANALYZE_ERROR: {exc}")
            lines.append("- CASE RESULT: FAIL")
            lines.append("")
            overall_fail += 1
            gc.collect()
            continue
        caption = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        )
        ve = result.verified_evidence
        packet = build_evidence_packet(
            verified_evidence=ve,
            canonical_caption_en=caption,
            evidence_brief=getattr(result, "evidence_brief", "") or "",
        )
        va = VisionAssistant()
        session = VisionAssistantSession(image_key=str(path), evidence=packet)
        answers: dict[str, str] = {}
        for q in questions:
            answers[q] = va.answer(session, q, language="en")

        confirmed = []
        if ve is not None:
            confirmed = [
                a.activity
                for a in ve.activities
                if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
            ]
        suggestions = generate_suggested_questions(packet, language="en", limit=4)

        checks = _checks(name, caption, answers, rules)
        fails = [k for k, v in checks.items() if v == "FAIL"]
        if fails:
            overall_fail += 1

        print(f"CAPTION: {caption}", flush=True)
        for q, a in answers.items():
            print(f"Q: {q}\nA: {a}", flush=True)
        print(f"CHECKS: {checks} RESULT={'FAIL' if fails else 'PASS'}", flush=True)

        lines.append(f"## IMAGE: {name}")
        lines.append(f"- Path: `{path}`")
        lines.append(f"- CAPTION: {caption}")
        lines.append(f"- CONFIRMED activities: {confirmed}")
        lines.append("- QA TESTS:")
        for q, a in answers.items():
            lines.append(f"  - {q} → {a}")
        lines.append("- SUGGESTED:")
        for s in suggestions:
            lines.append(f"  - {s}")
        lines.append("- FACT CHECK:")
        for k, v in checks.items():
            lines.append(f"  - {k}: {v}")
        lines.append(f"- CASE RESULT: {'FAIL' if fails else 'PASS'}")
        lines.append("")
        del result, packet, session, answers, ve
        gc.collect()

    lines.append(f"**Critical cases failed:** {overall_fail}/{len(CASES)}")
    out = Path("tmp/FINAL_COMPETITION_FREEZE_VALIDATION.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} critical_fails={overall_fail}", flush=True)


if __name__ == "__main__":
    main()
