"""Final polish real-image validation — caption naturalness gates only."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel

CASES = [
    ("soccer", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("farm", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("motorcycle", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("bicycle", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("baseball", Path("tmp/uploads/141139674_246c0f90a1.jpg")),
    ("kitchen", Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")),
    ("dense", Path("tmp/uploads/random_385406.jpg")),
    ("outdoor_misc", Path("tmp/uploads/47870024_73a4481f7d.jpg")),
]

OUT = ROOT / "tmp" / "FINAL_POLISH_VALIDATION.json"


def _caption(result) -> str:
    return (
        getattr(result.caption, "canonical_caption_en", None)
        or getattr(result.caption, "narrative_full", None)
        or result.caption.text
        or ""
    ).strip()


def _checks(name: str, caption: str, result) -> list[str]:
    fails: list[str] = []
    lower = caption.lower()
    ve = result.verified_evidence

    if re.search(r"\b(?:they|he|she)\s+are,\s+(?:a|an|the)\b", lower):
        fails.append("malformed_they_are_fragment")
    if re.search(
        r"\ba person,\s+\w+(?:\s*,\s*\w+)*\s+and\s+(?:riding|holding|leading|playing)\b",
        lower,
    ):
        fails.append("malformed_person_inventory_fragment")
    if "a person is riding. a person is riding" in lower:
        fails.append("duplicate_bare_riding")
    if lower.count("a person is riding.") >= 2:
        fails.append("repeated_bare_riding_sentence")
    if lower.count("holding a rope") + lower.count("holds a rope") >= 2:
        fails.append("duplicate_rope_hold")
    if re.search(r"\bthey\s+leads\b", lower):
        fails.append("grammar_they_leads")

    if name == "soccer":
        people_n = ve.people_count if ve else 0
        football_actors = 0
        if ve:
            for a in ve.activities:
                if a.evidence_level != ActivityEvidenceLevel.CONFIRMED:
                    continue
                if "football" in (a.activity or "").lower() or "soccer" in (a.activity or "").lower():
                    football_actors = max(
                        football_actors,
                        sum(1 for e in (a.entity_ids or ()) if str(e).startswith("person")),
                    )
        if "one person is playing football" in lower or "a person is playing football" in lower:
            fails.append("soccer_singular_activity_regression")
        if re.search(r"\b(?:four|4)\s+people\s+are\s+playing\b", lower):
            fails.append("soccer_actor_count_inflated_to_census")
        if football_actors == 2 and "two people are playing football" not in lower:
            # Allow close paraphrases with while-clause.
            if not re.search(r"\btwo people are playing football\b", lower):
                fails.append("soccer_missing_two_people_playing")
        if people_n >= 4 and "two people are playing" in lower:
            pass  # expected good

    if name == "motorcycle":
        if "on the water" in lower:
            fails.append("motorcycle_water_hallucination_fragment")
        if "riding" not in lower and "motorcycle" not in lower and "dirt bike" not in lower:
            fails.append("motorcycle_lost_activity")

    return fails


def main() -> int:
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
    fail_n = 0
    for name, path in CASES:
        if not path.exists():
            rows.append({"name": name, "path": str(path), "status": "MISSING"})
            fail_n += 1
            continue
        print(f"=== {name} {path.name} ===", flush=True)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        cap = _caption(result)
        ve = result.verified_evidence
        people = None if ve is None else ve.people_count
        acts = []
        if ve is not None:
            for a in ve.activities:
                if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe:
                    acts.append({"activity": a.activity, "actors": list(a.entity_ids)})
        fails = _checks(name, cap, result)
        status = "PASS" if not fails else "FAIL"
        if fails:
            fail_n += 1
        row = {
            "name": name,
            "path": str(path),
            "status": status,
            "fails": fails,
            "people_count": people,
            "activities": acts,
            "caption": cap,
        }
        rows.append(row)
        print(f"  {status} people={people} fails={fails}", flush=True)
        print(f"  CAPTION: {cap[:240]}", flush=True)

    payload = {"fails": fail_n, "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nTOTAL fails={fail_n} wrote {OUT}", flush=True)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
