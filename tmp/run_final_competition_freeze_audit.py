"""FINAL COMPETITION AUDIT — read-only. No production mutations."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.contracts.verified_evidence import ActivityEvidenceLevel

CASES = [
    ("soccer", "critical", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("farm", "critical", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("motorcycle", "critical", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("bicycle", "critical", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("baseball", "residual", Path("tmp/uploads/141139674_246c0f90a1.jpg")),
    ("kitchen", "general", Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")),
    ("moto_alt", "general", Path("tmp/uploads/166321294_4a5e68535f.jpg")),
    ("dense", "residual", Path("tmp/uploads/random_385406.jpg")),
    ("outdoor_misc", "general", Path("tmp/uploads/47870024_73a4481f7d.jpg")),
    ("trail", "general", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("animal", "general", Path("tmp/uploads/random_850976.jpg")),
]

OUT = ROOT / "tmp" / "FINAL_COMPETITION_FREEZE_AUDIT.json"
OUT_MD = ROOT / "tmp" / "FINAL_COMPETITION_FREEZE_AUDIT.md"

_PERSON = {"person", "man", "woman", "child", "people", "skier", "rider"}


def _cap(result) -> str:
    return (
        getattr(result.caption, "canonical_caption_en", None)
        or getattr(result.caption, "narrative_full", None)
        or result.caption.text
        or ""
    ).strip()


def _analyze(name: str, tier: str, path: Path, result) -> dict:
    ctx = result.scene_context
    ve = result.verified_evidence
    caption = _cap(result)
    lower = caption.lower()
    graph = Counter(n.label.lower() for n in ctx.graph.nodes)
    people_graph = graph.get("person", 0)
    people_ve = None if ve is None else ve.people_count

    confirmed = []
    if ve is not None:
        for a in ve.activities:
            if a.evidence_level != ActivityEvidenceLevel.CONFIRMED:
                continue
            if not a.narrative_safe:
                continue
            person_actors = [
                eid
                for eid in (a.entity_ids or ())
                if str(eid).startswith("person")
            ]
            confirmed.append(
                {
                    "activity": a.activity,
                    "entity_ids": list(a.entity_ids or ()),
                    "person_actors": person_actors,
                    "n_person_actors": len(person_actors),
                    "confidence": round(a.confidence, 3),
                }
            )

    rels = []
    if ve is not None:
        for r in ve.relations:
            if r.narrative_safe or r.qa_safe:
                rels.append(
                    {
                        "type": r.relation_type,
                        "subject": r.subject_id,
                        "object": r.object_id,
                        "conf": round(r.confidence, 3),
                    }
                )

    env = {}
    if ctx.environment is not None:
        env = {
            "scene_type": getattr(ctx.environment, "scene_type", None),
            "setting": getattr(ctx.environment, "setting", None),
            "indoor_outdoor": getattr(ctx.environment, "indoor_outdoor", None),
            "evidence": list(getattr(ctx.environment, "evidence", ()) or ())[:12],
        }

    flags: list[dict] = []

    def flag(sev: str, code: str, detail: str) -> None:
        flags.append({"severity": sev, "code": code, "detail": detail})

    # Generic caption health
    if re.search(r"\b(?:they|he|she)\s+are,\s+(?:a|an|the)\b", lower):
        flag("HIGH", "malformed_fragment", "Truncated 'they are, a …' fragment in caption")
    if re.search(
        r"\ba person,\s+\w+(?:\s*,\s*\w+)*\s+and\s+(?:riding|holding|leading|playing)\b",
        lower,
    ):
        flag("HIGH", "malformed_inventory", "Malformed person inventory fragment")
    if "a person is riding. a person is riding" in lower:
        flag("HIGH", "duplicate_riding", "Duplicate bare riding sentences")
    if lower.count("holding a rope") + lower.count("holds a rope") >= 2:
        flag("HIGH", "duplicate_rope", "Duplicate rope-hold claims")
    if re.search(r"\bthey\s+(?:leads|guides|moves|holds|wears|rides)\b", lower):
        flag("MEDIUM", "grammar_agreement", "Plural pronoun with singular verb")

    # Soccer-specific gates
    if name == "soccer":
        if people_graph < 4 and (people_ve or 0) < 4:
            flag("HIGH", "soccer_person_count", f"Expected ~4 people; graph={people_graph} ve={people_ve}")
        football = [a for a in confirmed if "football" in a["activity"].lower() or "soccer" in a["activity"].lower()]
        actors = max((a["n_person_actors"] for a in football), default=0)
        if actors and actors != 2:
            flag("MEDIUM", "soccer_actor_count_evidence", f"Football actors={actors} (expected 2)")
        if re.search(r"\b(?:four|4)\s+people\s+are\s+playing\b", lower):
            flag("CRITICAL", "soccer_census_as_actors", "Caption attributes play to global census")
        if "one person is playing football" in lower or re.search(
            r"\ba person is playing football\b", lower
        ):
            flag("CRITICAL", "soccer_singular_regression", "Singular football actor caption")
        if "two people are playing football" not in lower and "playing football" in lower:
            flag("HIGH", "soccer_missing_two", "Playing football without 'Two people'")
        if "two people are playing football" in lower:
            flag("INFO", "soccer_actor_ok", "Two-actor football caption preserved")

    # Farm
    if name == "farm":
        if people_graph <= 1:
            flag(
                "MEDIUM",
                "farm_second_person_yolo_miss",
                "Only 1 person detected at imgsz=1280 (known YOLO miss; freeze decision accepted)",
            )
        if lower.count("holding a rope") + lower.count("holds a rope") >= 2:
            flag("HIGH", "farm_rope_dup", "Rope claim duplicated")
        if "leading" in lower and ("rope" in lower or "holding" in lower or "holds" in lower):
            flag("INFO", "farm_dual_activity_ok", "Leading + rope expressed without obvious dup")

    # Motorcycle water
    if name == "motorcycle":
        if "water" in lower:
            flag(
                "HIGH",
                "motorcycle_water_hallucination",
                "Caption mentions water — verify visual support; likely VLM hallucination if unsupported",
            )
        if "riding" in lower or "dirt bike" in lower or "motorcycle" in lower:
            flag("INFO", "motorcycle_activity_present", "Riding/motorcycle retained")

    # Bicycle riding vs confirmed
    if name == "bicycle":
        has_bike = graph.get("bicycle", 0) > 0
        riding = [a for a in confirmed if "rid" in a["activity"].lower()]
        if has_bike and not riding and "riding" not in lower:
            flag(
                "MEDIUM",
                "bicycle_riding_not_confirmed",
                "Bicycle present; riding not CONFIRMED / not in caption (frozen activity system)",
            )
        if "riding" in lower and not riding:
            flag("LOW", "bicycle_riding_caption_without_confirmed", "Caption says riding without CONFIRMED activity")

    # Baseball fragments
    if name == "baseball":
        if "they are," in lower:
            flag("HIGH", "baseball_malformed", "Malformed VLM fragment remains")
        if "swinging" in lower or "bat" in lower:
            flag("INFO", "baseball_core_ok", "Core bat/swing content present")

    # Dense over-attribution
    if name == "dense":
        if people_ve and people_ve >= 6:
            for a in confirmed:
                n = a["n_person_actors"]
                # Caption uses global-ish people count for activity
                m = re.search(
                    rf"\b({people_ve}|twelve|12|six|6|ten|10)\s+people\b[^.]*\b{re.escape(a['activity'].split()[0])}",
                    lower,
                )
                if n and n < (people_ve or 0) and m:
                    flag(
                        "HIGH",
                        "dense_overattribution",
                        f"Caption may attribute activity to census while actors={n}/{people_ve}",
                    )
            if re.search(r"\b(?:all|everyone|each person)\b[^.]*\b(?:playing|holding)\b", lower):
                # 'each person holds' can be OK if verified; mark MEDIUM for review
                flag(
                    "MEDIUM",
                    "dense_each_person_claim",
                    "Caption uses each/all-person activity wording — verify against actor subset",
                )

    # Generic subset inflation
    for a in confirmed:
        n = a["n_person_actors"]
        if n >= 1 and people_ve and people_ve > n:
            # Look for "<global count> people are <activity>"
            act_head = a["activity"].split()[0]
            if re.search(
                rf"\b(?:four|five|six|seven|eight|nine|ten|twelve|\d+)\s+people\s+are\s+{re.escape(act_head)}",
                lower,
            ):
                # Extract number
                flag(
                    "HIGH",
                    "subset_activity_inflated",
                    f"Possible census applied to activity '{a['activity']}' (actors={n}, people={people_ve})",
                )

    qr = result.quality_report
    q = None
    if qr is not None:
        for attr in ("overall_score", "overall", "score"):
            if hasattr(qr, attr) and getattr(qr, attr) is not None:
                q = float(getattr(qr, attr))
                break

    return {
        "name": name,
        "tier": tier,
        "path": str(path),
        "graph_labels": dict(graph),
        "people_graph": people_graph,
        "people_ve": people_ve,
        "confirmed_activities": confirmed,
        "relations": rels[:12],
        "environment": env,
        "caption": caption,
        "quality_overall": q,
        "qa_passed": bool(result.qa_passed),
        "flags": flags,
        "worst": min((f["severity"] for f in flags), default="INFO", key=lambda s: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"INFO":4}.get(s, 9)),
    }


def main() -> int:
    images = [(n, t, p) for n, t, p in CASES if p.exists()]
    print(f"AUDIT images={len(images)}", flush=True)
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
    for name, tier, path in images:
        print(f"\n=== {name} {path.name} ===", flush=True)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        row = _analyze(name, tier, path, result)
        rows.append(row)
        print(
            f"  people_graph={row['people_graph']} ve={row['people_ve']} "
            f"acts={[a['activity'] for a in row['confirmed_activities']]} "
            f"worst={row['worst']} flags={len(row['flags'])}",
            flush=True,
        )
        print(f"  CAPTION: {row['caption'][:220]}", flush=True)

    payload = {"n": len(rows), "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
