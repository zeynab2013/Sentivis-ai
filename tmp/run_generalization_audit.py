"""GENERALIZATION AUDIT — read-only, no production mutations.

Runs a diverse real-image matrix and flags semantic ownership failure classes A–L.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Diverse matrix covering requested categories (existing uploads only).
CASES: list[tuple[str, str, Path]] = [
    ("1_single_person_activity", "motorcycle rider", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("2_multi_shared_activity", "soccer shared play", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("3_multi_different_activities", "farm leading+holding", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("4_sports_soccer_alt", "football sports alt", Path("tmp/uploads/141139674_246c0f90a1.jpg")),
    ("5_person_animal", "horse interaction", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("6_multi_animal", "two horses+fire", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("7_person_vehicle_bike", "bicycle multi people", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("7b_person_vehicle_moto_alt", "motorcycle alt", Path("tmp/uploads/166321294_4a5e68535f.jpg")),
    ("8_indoor_kitchen", "kitchen objects", Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")),
    ("9_outdoor_trail", "outdoor landscape/trail", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("10_group_sports", "soccer group", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("11_many_objects", "kitchen dense objects", Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")),
    ("12_weak_activity", "animal/no-strong-human-act", Path("tmp/uploads/random_850976.jpg")),
    ("13_overlap_people", "soccer overlap", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("14_subset_participants", "soccer 2 of N playing", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("15_simultaneous_activities", "farm dual acts", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("16_dense_indoor", "dense indoor/group", Path("tmp/uploads/random_385406.jpg")),
    ("17_outdoor_misc", "misc outdoor", Path("tmp/uploads/47870024_73a4481f7d.jpg")),
]

OUT_JSON = ROOT / "tmp" / "GENERALIZATION_AUDIT.json"
OUT_MD = ROOT / "tmp" / "GENERALIZATION_AUDIT.md"

_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "a": 1,
    "an": 1,
}


def _qty(token: str) -> int | None:
    t = (token or "").lower().strip()
    if t.isdigit():
        return int(t)
    return _NUM.get(t)


def _analyze(name: str, category: str, path: Path, result) -> dict:
    from core.contracts.verified_evidence import ActivityEvidenceLevel

    ctx = result.scene_context
    ve = result.verified_evidence
    qr = result.quality_report
    caption = (
        getattr(result.caption, "canonical_caption_en", None)
        or result.caption.narrative_full
        or result.caption.text
        or ""
    ).strip()
    cap_l = caption.lower()

    graph_labels = Counter(n.label.lower() for n in ctx.graph.nodes)
    person_nodes = [n for n in ctx.graph.nodes if n.label.lower() == "person"]
    people_n = 0 if ve is None else ve.people_count

    activities = []
    actor_map: dict[str, list[str]] = {}
    if ve is not None:
        for a in ve.activities:
            if a.evidence_level != ActivityEvidenceLevel.CONFIRMED or not a.narrative_safe:
                continue
            persons = []
            for eid in a.entity_ids:
                ent = ve.entity_by_id(eid)
                if ent is not None and ent.label.lower() in {
                    "person",
                    "man",
                    "woman",
                    "child",
                    "people",
                }:
                    persons.append(eid)
            activities.append(
                {
                    "activity": a.activity,
                    "entity_ids": list(a.entity_ids),
                    "person_actors": persons,
                    "n_person_actors": len(persons),
                    "confidence": a.confidence,
                }
            )
            actor_map[a.activity] = persons

    relations = []
    if ve is not None:
        for r in ve.narrative_relations():
            relations.append(
                {
                    "type": r.relation_type,
                    "subject": r.subject_id,
                    "object": r.object_id,
                }
            )

    env = {
        "indoor_outdoor": getattr(ctx.environment, "indoor_outdoor", ""),
        "setting": getattr(ctx.environment, "setting", ""),
        "scene_type": getattr(ctx.environment, "scene_type", ""),
    }
    if ve is not None:
        env = {
            "indoor_outdoor": ve.scene.indoor_outdoor,
            "setting": ve.scene.setting,
            "scene_type": ve.scene.scene_type,
        }

    # Caption mentions of people+action
    mentions = []
    for m in re.finditer(
        r"\b((?:one|two|three|four|five|six|\d+|a|an)\s+(?:people|person)|a\s+person|one\s+person)\s+"
        r"(is|are)\s+([a-z]+(?:\s+[a-z]+){0,6})",
        cap_l,
    ):
        mentions.append(
            {
                "subject": m.group(1),
                "qty": _qty(m.group(1).split()[0]),
                "predicate": f"{m.group(2)} {m.group(3)}".strip(),
            }
        )

    failures: list[dict] = []

    def fail(code: str, severity: str, detail: str, stage: str) -> None:
        failures.append(
            {
                "code": code,
                "severity": severity,
                "detail": detail,
                "likely_stage": stage,
            }
        )

    # A / C / G: global census applied to activity subject
    for act in activities:
        n_act = act["n_person_actors"]
        tok0 = (act["activity"] or "").split()[0]
        if n_act >= 1 and people_n > n_act:
            if re.search(
                rf"\b(?:{people_n}|four|three|five|six)\s+people\s+are\s+{re.escape(tok0)}",
                cap_l,
            ):
                fail(
                    "A/C/G",
                    "CRITICAL",
                    f"Caption assigns '{act['activity']}' to all {people_n} people; "
                    f"verified actors={n_act} ({act['person_actors']})",
                    "count clamping / coverage",
                )
        # B: singular activity inflated? hard without GT; flag if caption says many for 1-actor act
        if n_act == 1 and re.search(
            rf"\b(?:two|three|four|five|\d+)\s+people\s+are\s+{re.escape(tok0)}",
            cap_l,
        ):
            fail(
                "B",
                "HIGH",
                f"Single-actor activity '{act['activity']}' captioned as multi-person",
                "coverage / clamp",
            )
        # shared act collapsed to one person
        if n_act >= 2 and re.search(
            rf"\b(?:one|a)\s+person\s+is\s+{re.escape(tok0)}",
            cap_l,
        ):
            fail(
                "C",
                "CRITICAL",
                f"Multi-actor activity '{act['activity']}' collapsed to singular person",
                "coverage / filtering",
            )

    # D: distinct activities merged into one subject count incorrectly —
    # if >=2 confirmed person activities with different actors and caption only names one activity once for all people
    if len(activities) >= 2:
        actor_sets = [tuple(a["person_actors"]) for a in activities if a["n_person_actors"] >= 1]
        if len({s for s in actor_sets if s}) >= 2:
            # both activities should appear if coverage injected them; if only one appears, MEDIUM
            present = sum(1 for a in activities if a["activity"].split()[0] in cap_l)
            if present < min(2, len(activities)):
                fail(
                    "D",
                    "MEDIUM",
                    "Multiple distinct person activities in evidence; caption missing some",
                    "filtering / arbitration / coverage",
                )

    # E: object count confused with actor — e.g. "4 sports balls are playing" unlikely; check horse/people mix
    if re.search(r"\b\d+\s+(?:horses|bicycles|motorcycles|balls)\s+are\s+(?:playing|riding|leading|holding)\b", cap_l):
        fail("E", "HIGH", "Object class used as activity subject quantity", "clamp / coverage")

    # I: robotic census-only when rich evidence exists
    robotic = bool(
        re.fullmatch(
            r"(?:\d+|two|three|four|five)\s+people\s+are\s+visible\.?(?:\s+a\s+[^.]+)?",
            cap_l.strip(),
        )
    )
    if robotic and activities:
        fail(
            "I",
            "MEDIUM",
            "Caption reduced to robotic census despite confirmed activities",
            "filtering / arbitration",
        )

    # thin soccer-style stub: activity + object only, dropping rich clothing when colors exist
    # (informational — not always a failure)
    natural_notes = []
    if len(caption.split()) < 18 and people_n >= 2 and activities:
        natural_notes.append("thin caption for multi-person scene")
    if re.search(r"\b\d+\s+people\s+are\s+visible\b", cap_l) and re.search(
        r"\b\d+\s+people\s+are\s+(?:playing|riding|holding|leading)\b", cap_l
    ):
        natural_notes.append("census + activity split (acceptable if quantities correct)")

    # K / L: unsupported tokens vs confirmed activities presence
    hall = None if qr is None else qr.hallucination_risk
    if hall is not None and hall >= 0.25:
        fail("K", "MEDIUM", f"Hallucination risk elevated ({hall})", "claim filtering / generation")
    for a in activities:
        # activity token missing entirely
        key = a["activity"].lower()
        tokens = [t for t in key.split() if t not in {"a", "an", "the", "with"}]
        if tokens and not any(t in cap_l for t in tokens[:2]):
            fail(
                "L",
                "HIGH",
                f"Confirmed activity '{a['activity']}' absent from final caption",
                "filtering / arbitration",
            )

    # Environment over-inference heuristic
    io = (env.get("indoor_outdoor") or "").lower()
    if "restaurant" in cap_l and "kitchen" in (env.get("setting") or "").lower():
        fail("J", "MEDIUM", "Restaurant claimed while kitchen setting evidenced", "environment / VLM")
    if "classroom" in cap_l and "classroom" not in json.dumps(env).lower():
        fail("J", "LOW", "Classroom mentioned without env support", "environment / VLM")

    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    worst = None
    if failures:
        worst = sorted(failures, key=lambda f: severity_rank.get(f["severity"], 9))[0]["severity"]

    return {
        "id": name,
        "category": category,
        "path": str(path),
        "detected_objects": dict(graph_labels),
        "scene_graph_nodes": len(ctx.graph.nodes),
        "person_nodes": len(person_nodes),
        "verified_people_count": people_n,
        "verified_activities": activities,
        "verified_relations": relations,
        "environment": env,
        "final_caption": caption,
        "caption_actor_mentions": mentions,
        "actor_ownership_ok": not any(f["code"] in {"A/C/G", "B", "C"} for f in failures),
        "object_counts_ok": "E" not in {f["code"] for f in failures},
        "natural_notes": natural_notes,
        "hallucination_risk": hall,
        "evidence_consistency": None if qr is None else qr.evidence_consistency,
        "object_coverage": None if qr is None else qr.object_coverage,
        "activity_coverage": None if qr is None else qr.activity_coverage,
        "relationship_coverage": None if qr is None else qr.relationship_coverage,
        "overall_quality": None if qr is None else qr.overall_quality,
        "failures": failures,
        "worst_severity": worst,
        "status": "FAIL" if any(f["severity"] in {"CRITICAL", "HIGH"} for f in failures) else (
            "ISSUE" if failures else "PASS"
        ),
    }


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest

    # Deduplicate paths while keeping first category label for each unique path in run list;
    # still report every category row (reusing cached result per path).
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )

    cache: dict[str, object] = {}
    rows: list[dict] = []
    for name, category, rel in CASES:
        path = ROOT / rel
        print(f"=== {name} | {category} | {path.name} ===", flush=True)
        if not path.is_file():
            rows.append(
                {
                    "id": name,
                    "category": category,
                    "path": str(path),
                    "status": "MISSING",
                    "failures": [],
                    "final_caption": "",
                }
            )
            continue
        key = str(path.resolve())
        if key not in cache:
            cache[key] = orch.analyze(PipelineRequest(image_path=path, options=opts))
        row = _analyze(name, category, path, cache[key])
        rows.append(row)
        print(f"STATUS={row['status']} caption={row['final_caption'][:160]}", flush=True)
        for f in row["failures"]:
            print(f"  [{f['severity']}] {f['code']}: {f['detail']}", flush=True)

    critical = [r for r in rows if r.get("worst_severity") == "CRITICAL"]
    high = [r for r in rows if r.get("worst_severity") == "HIGH"]
    issues = [r for r in rows if r.get("status") == "ISSUE"]
    fails = [r for r in rows if r.get("status") == "FAIL"]
    missing = [r for r in rows if r.get("status") == "MISSING"]

    if critical or high or fails:
        verdict = "FAIL" if critical or high or fails else "PASS WITH ISSUES"
    elif issues:
        verdict = "PASS WITH ISSUES"
    else:
        verdict = "PASS"

    # Unique-path summary for the table
    payload = {
        "verdict": verdict,
        "summary": {
            "rows": len(rows),
            "unique_images": len(cache),
            "fail": len(fails),
            "issue": len(issues),
            "missing": len(missing),
            "critical_rows": len(critical),
            "high_rows": len(high),
        },
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# GENERALIZATION AUDIT",
        "",
        f"**VERDICT: {verdict}**",
        "",
        f"Unique images analyzed: {len(cache)} | Category rows: {len(rows)}",
        f"FAIL rows: {len(fails)} | ISSUE rows: {len(issues)} | MISSING: {len(missing)}",
        "",
        "| ID | Category | People | Activities (actors) | Caption (short) | Status | Failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        acts = "; ".join(
            f"{a['activity']}[{a['n_person_actors']}]" for a in r.get("verified_activities", [])
        ) or "—"
        cap = (r.get("final_caption") or "").replace("|", "/")
        if len(cap) > 90:
            cap = cap[:87] + "..."
        fails_s = "; ".join(
            f"{f['code']}:{f['severity']}" for f in r.get("failures", [])
        ) or "—"
        lines.append(
            f"| {r['id']} | {r.get('category','')} | {r.get('verified_people_count','')} | {acts} | {cap} | {r.get('status')} | {fails_s} |"
        )

    lines.extend(["", "## Detailed failures", ""])
    any_fail = False
    for r in rows:
        if not r.get("failures"):
            continue
        any_fail = True
        lines.append(f"### {r['id']} ({r.get('category')})")
        lines.append(f"- Path: `{r.get('path')}`")
        lines.append(f"- Caption: {r.get('final_caption')}")
        lines.append(f"- People: {r.get('verified_people_count')}")
        lines.append(f"- Activities: `{json.dumps(r.get('verified_activities'))}`")
        lines.append(f"- Relations: `{json.dumps(r.get('verified_relations'))}`")
        lines.append(f"- Environment: `{json.dumps(r.get('environment'))}`")
        for f in r["failures"]:
            lines.append(
                f"- **{f['severity']} [{f['code']}]** {f['detail']} "
                f"(likely stage: {f['likely_stage']})"
            )
        lines.append("")
    if not any_fail:
        lines.append("No automated failure-class hits.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Audit is evidence-vs-caption based; it does not use external human GT labels.",
            "- Naturalness notes are advisory and do not alone cause FAIL.",
            "- Duplicate category rows reuse the same pipeline result for the same image path.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nVERDICT={verdict}", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
