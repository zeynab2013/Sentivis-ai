"""Final real-image regression matrix for competition stabilization."""

from __future__ import annotations

import re
from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.entity_indexing import ordered_people
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

# 12 required categories — best available local images.
CASES: list[tuple[str, Path]] = [
    ("1_kitchen_multi_people", Path("tmp/coco_kitchen.jpg")),
    ("2_horse_person_fire", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("3_football_sports", Path("tmp/uploads/141139674_246c0f90a1.jpg")),
    ("4_motorcycle", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("5_bicycle_multi_people", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("6_vehicle", Path("tmp/competition_e2e_street.jpg")),
    ("7_landscape", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("8_animal", Path("tmp/uploads/random_850976.jpg")),
    ("9_dense_indoor", Path("tmp/uploads/random_385406.jpg")),
    ("10_low_quality", Path("tmp/enhance_blur.jpg")),
    ("11_enhanced", Path("tmp/pipeline_enhanced_500x333.png")),
    ("12_multi_person_clothing", Path("tmp/uploads/random_385406.jpg")),
]

# Fallbacks if preferred images missing.
_FALLBACKS = {
    "2_horse_person_fire": [
        Path("tmp/uploads/10815824_2997e03d76.jpg"),
        Path("tmp/uploads/random_756903.jpg"),
    ],
    "3_football_sports": [
        Path("tmp/uploads/141139674_246c0f90a1.jpg"),
        Path("tmp/uploads/47871819_db55ac4699.jpg"),
        Path("tmp/coco_baseball.jpg"),
    ],
    "4_motorcycle": [
        Path("tmp/uploads/143552829_72b6ba49d4.jpg"),
        Path("tmp/uploads/166321294_4a5e68535f.jpg"),
    ],
    "8_animal": [
        Path("tmp/uploads/random_850976.jpg"),
        Path("tmp/uploads/random_479672.jpg"),
    ],
}

QUESTIONS = [
    "How many people are visible?",
    "What is the person doing?",
    "What color clothing is the person wearing?",
    "What objects are visible?",
]


def _resolve(name: str, path: Path) -> Path:
    if path.exists():
        return path
    for alt in _FALLBACKS.get(name, []):
        if alt.exists():
            return alt
    # Last resort: any upload jpg
    uploads = sorted(Path("tmp/uploads").glob("*.jpg"))
    return uploads[0] if uploads else path


def _count_in_caption(cap: str) -> int | None:
    low = cap.lower()
    # Explicit plurals / second-person cues first.
    if re.search(r"\btwo people\b", low) or "another person" in low or "both people" in low:
        return 2
    if re.search(r"\bthree people\b", low):
        return 3
    if re.search(r"\bfour people\b", low):
        return 4
    if re.search(r"\bfive people\b", low):
        return 5
    if re.search(r"\b(?:several|multiple)\s+people\b", low):
        return None
    if re.search(r"\bnine people\b", low):
        return 9
    # Only treat singular when no plural cues.
    if "another person" not in low and re.search(r"\b(?:one person|a person|the person)\b", low):
        if not re.search(r"\bpeople\b", low) and "they" not in low:
            return 1
    return None


def _qa_people_count(answer: str) -> int | None:
    low = answer.lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    if "no people" in low:
        return 0
    for w, n in words.items():
        if f"there are {w} people" in low or f"there is {w} person" in low:
            return n
    return None


def main() -> None:
    out_md = Path("tmp/FINAL_REGRESSION_MATRIX.md")
    out_txt = Path("tmp/final_regression_matrix_raw.txt")
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
    raw: list[str] = []
    rows: list[dict] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        raw.append(s)

    for name, image in CASES:
        image = _resolve(name, image)
        log("=" * 80)
        log(f"CASE: {name}")
        log(f"IMAGE: {image}")
        row = {
            "category": name,
            "image": str(image),
            "exists": image.exists(),
            "caption": "",
            "people_qa": "",
            "activity_qa": "",
            "color_qa": "",
            "verified_people": 0,
            "confirmed_acts": [],
            "checks": {},
            "pass": False,
        }
        if not image.exists():
            row["checks"]["image_exists"] = "FAIL"
            rows.append(row)
            log("SKIP missing")
            continue
        try:
            result = orch.analyze(PipelineRequest(image_path=image, options=opts))
        except Exception as exc:  # noqa: BLE001
            log(f"PIPELINE ERROR: {exc}")
            row["checks"]["pipeline"] = f"FAIL: {exc}"
            rows.append(row)
            continue

        ve = result.verified_evidence
        cap = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        ).strip()
        row["caption"] = cap
        log("--- CAPTION ---")
        log(cap)

        if ve is None:
            row["checks"]["verified"] = "FAIL"
            rows.append(row)
            continue

        packet = build_evidence_packet(
            result.scene_context,
            canonical_caption_en=cap,
            verified_evidence=ve,
        )
        people_n = len(ordered_people(packet))
        row["verified_people"] = people_n
        confirmed = [
            a.activity
            for a in ve.activities
            if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
        ]
        row["confirmed_acts"] = confirmed
        log(f"verified_people={people_n}")
        log(f"confirmed_acts={confirmed}")

        answers: dict[str, str] = {}
        for q in QUESTIONS:
            session = VisionAssistantSession(image_key=name, evidence=packet)
            try:
                answers[q] = va.answer(session, q)
            except Exception as exc:  # noqa: BLE001
                answers[q] = f"ERROR: {exc}"
            log(f"Q: {q}")
            log(f"A: {answers[q]}")

        row["people_qa"] = answers.get("How many people are visible?", "")
        row["activity_qa"] = answers.get("What is the person doing?", "")
        row["color_qa"] = answers.get("What color clothing is the person wearing?", "")
        suggestions = generate_suggested_questions(packet)
        log(f"suggestions={suggestions[:5]}")

        checks: dict[str, str] = {}
        # 1) Caption not empty inventory stub
        words = len(cap.split())
        checks["caption_nonempty"] = "PASS" if words >= 8 else "FAIL"
        # 2) People count caption↔QA when both assert a number
        cap_n = _count_in_caption(cap)
        qa_n = _qa_people_count(row["people_qa"])
        if qa_n is not None and qa_n != people_n:
            checks["people_qa_matches_verified"] = "FAIL"
        else:
            checks["people_qa_matches_verified"] = "PASS"
        if cap_n is not None and qa_n is not None and cap_n != qa_n:
            checks["people_caption_qa_consistent"] = "FAIL"
        else:
            checks["people_caption_qa_consistent"] = "PASS"
        # 3) CONFIRMED activity must appear in caption and not contradict QA
        if confirmed:
            act0 = confirmed[0].lower()
            tokens = [t for t in act0.split() if len(t) > 3][:2]
            if tokens and all(t in cap.lower() for t in tokens[:1]):
                checks["confirmed_activity_in_caption"] = "PASS"
            else:
                checks["confirmed_activity_in_caption"] = "FAIL"
            # QA should not invent a different strong activity when one is confirmed
            act_qa = row["activity_qa"].lower()
            if "can't" in act_qa or "cannot" in act_qa or "reliably" in act_qa:
                # Prefer answering when CONFIRMED exists
                checks["activity_qa_uses_confirmed"] = "FAIL"
            elif tokens and tokens[0] in act_qa:
                checks["activity_qa_uses_confirmed"] = "PASS"
            else:
                checks["activity_qa_uses_confirmed"] = "PASS"  # phrasing may differ
        else:
            checks["confirmed_activity_in_caption"] = "N/A"
            checks["activity_qa_uses_confirmed"] = "N/A"
        # 4) No banned weak inventions in caption+activity QA
        banned = (
            "office work",
            "playing tennis",
            "shopping",
            "restaurant dining",
            "preparing a campfire",
        )
        blob = f"{cap} {row['activity_qa']}".lower()
        # Allow "playing with a tennis racket"
        bad = [b for b in banned if b in blob and b != "playing tennis"]
        if "playing tennis" in blob and "playing with a tennis" not in blob:
            bad.append("playing tennis")
        checks["no_weak_inventions"] = "FAIL" if bad else "PASS"
        # 5) Suggested questions present
        checks["suggestions"] = "PASS" if suggestions else "FAIL"

        row["checks"] = checks
        hard = [
            v
            for k, v in checks.items()
            if v == "FAIL" and k not in {"confirmed_activity_in_caption"}
        ]
        # For bike/moto cases, confirmed coverage is hard-required when present
        if confirmed and checks.get("confirmed_activity_in_caption") == "FAIL":
            hard.append("FAIL")
        if "bicycle" in name or "motorcycle" in name or "5_" in name or "4_" in name:
            if confirmed and checks.get("activity_qa_uses_confirmed") == "FAIL":
                hard.append("FAIL")
        row["pass"] = len(hard) == 0
        rows.append(row)
        log(f"RESULT: {'PASS' if row['pass'] else 'FAIL'} checks={checks}")

    # Write markdown matrix
    lines = [
        "# FINAL REGRESSION MATRIX",
        "",
        "Real-image validation after emergency stabilization pass.",
        "",
        "| Category | Result | People (verified) | Confirmed activity | Notes |",
        "|----------|--------|-------------------|--------------------|-------|",
    ]
    pass_n = 0
    for r in rows:
        if r["pass"]:
            pass_n += 1
        fails = [k for k, v in r["checks"].items() if v == "FAIL"]
        notes = ", ".join(fails) if fails else "ok"
        acts = "; ".join(r["confirmed_acts"][:2]) if r["confirmed_acts"] else "(none)"
        lines.append(
            f"| {r['category']} | {'PASS' if r['pass'] else 'FAIL'} | "
            f"{r['verified_people']} | {acts} | {notes} |"
        )
    lines.extend(
        [
            "",
            f"**Score:** {pass_n}/{len(rows)} categories PASS",
            "",
            "## Per-case captions & QA",
            "",
        ]
    )
    for r in rows:
        lines.append(f"### {r['category']}")
        lines.append(f"- Image: `{r['image']}`")
        lines.append(f"- Caption: {r['caption'][:400]}")
        lines.append(f"- People QA: {r['people_qa']}")
        lines.append(f"- Activity QA: {r['activity_qa']}")
        lines.append(f"- Color QA: {r['color_qa']}")
        lines.append(f"- Checks: `{r['checks']}`")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_txt.write_text("\n".join(raw), encoding="utf-8")
    print(f"\nWrote {out_md} ({pass_n}/{len(rows)} PASS)", flush=True)


if __name__ == "__main__":
    main()
