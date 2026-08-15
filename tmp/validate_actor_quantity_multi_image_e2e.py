"""Multi-image real pipeline smoke after actor-quantity clamp fix.

Validation-only. Does not modify production code.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    ("SOCCER", ROOT / "tmp" / "uploads" / "47871819_db55ac4699.jpg"),
    ("HORSE", ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg"),
    ("MOTORCYCLE", ROOT / "tmp" / "uploads" / "143552829_72b6ba49d4.jpg"),
    ("BICYCLE", ROOT / "tmp" / "uploads" / "191003284_1025b0fb7d.jpg"),
]
OUT = ROOT / "tmp" / "actor_quantity_multi_image_e2e.json"


def _person_actor_counts(verified) -> dict[str, int]:
    from core.contracts.verified_evidence import ActivityEvidenceLevel

    out: dict[str, int] = {}
    if verified is None:
        return out
    for act in verified.activities:
        if act.evidence_level != ActivityEvidenceLevel.CONFIRMED or not act.narrative_safe:
            continue
        n = 0
        for eid in act.entity_ids:
            ent = verified.entity_by_id(eid)
            if ent is not None and ent.label.lower() in {
                "person",
                "man",
                "woman",
                "child",
                "people",
            }:
                n += 1
        out[act.activity] = n
    return out


def _caption_actor_mentions(caption: str) -> list[tuple[str, str]]:
    """Return (quantity_phrase, following_verb_phrase) for people+action patterns."""
    found = []
    for m in re.finditer(
        r"\b((?:one|two|three|four|five|six|\d+)\s+people|a\s+person|one\s+person)\s+"
        r"(is|are)\s+([a-z]+(?:\s+[a-z]+){0,6})",
        caption.lower(),
    ):
        found.append((m.group(1), f"{m.group(2)} {m.group(3)}".strip()))
    return found


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from core.contracts.verified_evidence import ActivityEvidenceLevel

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
    for name, path in CASES:
        if not path.is_file():
            rows.append({"name": name, "path": str(path), "status": "MISSING"})
            fails += 1
            continue
        print(f"=== {name} {path} ===", flush=True)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        caption = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.narrative_full
            or result.caption.text
            or ""
        ).strip()
        ve = result.verified_evidence
        people_n = 0 if ve is None else ve.people_count
        actor_counts = _person_actor_counts(ve)
        mentions = _caption_actor_mentions(caption)
        qr = result.quality_report

        # Detect global-census overwrite: caption attributes activity to all people
        # when verified actors are fewer.
        bad = False
        reasons = []
        for act, n_actors in actor_counts.items():
            if n_actors <= 0:
                continue
            if people_n > n_actors:
                # Look for "<people_n> people are <activity-ish>"
                act_tok = act.split()[0]  # playing / riding / leading / holding
                if re.search(
                    rf"\b(?:{people_n}|four|three|five|six)\s+people\s+are\s+{re.escape(act_tok)}",
                    caption,
                    re.I,
                ):
                    bad = True
                    reasons.append(
                        f"global census applied to activity '{act}' "
                        f"(actors={n_actors}, people={people_n})"
                    )
            if n_actors >= 2 and re.search(
                rf"\b(?:one|a)\s+person\s+is\s+{re.escape(act.split()[0])}",
                caption,
                re.I,
            ):
                # Shared activity collapsed to singular — only flag for playing*
                if act.lower().startswith("playing"):
                    bad = True
                    reasons.append(f"shared activity '{act}' collapsed to singular person")

        if name == "SOCCER":
            if re.search(r"\b(?:4|four)\s+people\s+are\s+playing\b", caption, re.I):
                bad = True
                reasons.append("soccer: 4 people are playing")
            if not re.search(r"\btwo\s+people\s+are\s+playing\b", caption, re.I):
                bad = True
                reasons.append("soccer: missing 'Two people are playing'")

        status = "FAIL" if bad else "PASS"
        if bad:
            fails += 1

        row = {
            "name": name,
            "path": str(path),
            "status": status,
            "reasons": reasons,
            "final_caption": caption,
            "verified_people_count": people_n,
            "verified_activity_actor_counts": actor_counts,
            "caption_actor_mentions": mentions,
            "quality": {
                "overall": None if qr is None else qr.overall_quality,
                "object_coverage": None if qr is None else qr.object_coverage,
                "activity_coverage": None if qr is None else qr.activity_coverage,
                "relationship_coverage": None if qr is None else qr.relationship_coverage,
                "evidence_consistency": None if qr is None else qr.evidence_consistency,
                "hallucination_risk": None if qr is None else qr.hallucination_risk,
            },
            "confirmed_activities": [
                a.activity
                for a in (ve.activities if ve else ())
                if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
            ],
        }
        rows.append(row)
        print(f"CAPTION: {caption}", flush=True)
        print(f"people={people_n} actors={actor_counts} STATUS={status}", flush=True)
        for r in reasons:
            print(f"  - {r}", flush=True)

    payload = {"results": rows, "fails": fails}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
