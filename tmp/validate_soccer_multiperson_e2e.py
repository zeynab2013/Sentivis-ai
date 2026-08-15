"""REAL soccer E2E validation — no production code mutations.

Matches Streamlit/competition freeze entry: StartupOrchestrator().run() →
pipeline._orchestrator.analyze(...). Temporary wrappers are script-local only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE = ROOT / "tmp" / "uploads" / "47871819_db55ac4699.jpg"
OUT = ROOT / "tmp" / "soccer_multiperson_e2e_result.json"

TRACE: dict = {
    "yolo_raw": [],
    "yolo_kept": [],
    "yolo_removed": [],
    "vlm_bridge": [],
    "caption_stages": {},
}


def main() -> int:
    if not IMAGE.is_file():
        print(f"FAIL: soccer image missing: {IMAGE}")
        return 2

    import analysis.evidence.vlm_activity_bridge as bridge
    import language.refinement.caption_coverage as cov
    import language.refinement.caption_sanity as sanity
    import language.validation.caption_factuality as fact
    import vision.detection.yolo_engine as yolo_mod

    _orig_ensure = cov.ensure_salient_verified_coverage
    _orig_filter = fact.filter_unsupported_claims_verified
    _orig_extract = bridge.extract_vlm_activity_candidates
    _orig_sanitize = sanity.sanitize_caption
    _orig_filter_dets = yolo_mod.YoloEngine._filter_detections

    def _wrap_ensure(text, **kwargs):
        before = (text or "").strip()
        out = _orig_ensure(text, **kwargs)
        TRACE["caption_stages"].setdefault("ensure_salient", []).append(
            {"before": before, "after": (out or "").strip()}
        )
        return out

    def _wrap_filter(text, verified):
        before = (text or "").strip()
        out = _orig_filter(text, verified)
        TRACE["caption_stages"].setdefault("filter_unsupported", []).append(
            {"before": before, "after": (out or "").strip()}
        )
        return out

    def _wrap_sanitize(text):
        before = (text or "").strip()
        out = _orig_sanitize(text)
        TRACE["caption_stages"].setdefault("sanitize", []).append(
            {"before": before, "after": (out or "").strip()}
        )
        return out

    def _wrap_extract(vlm_text, scene_context, **kwargs):
        cands = _orig_extract(vlm_text, scene_context, **kwargs)
        labels = {n.index: n.label.lower() for n in scene_context.graph.nodes}
        areas = {
            n.index: float(n.bounding_box_area_ratio) for n in scene_context.graph.nodes
        }
        for c in cands:
            TRACE["vlm_bridge"].append(
                {
                    "activity": c.activity,
                    "confidence": c.confidence,
                    "supporting_node_indices": list(c.supporting_node_indices),
                    "supporting_labels": [labels.get(i, "?") for i in c.supporting_node_indices],
                    "supporting_areas": [areas.get(i, 0.0) for i in c.supporting_node_indices],
                    "supporting_relation_types": list(c.supporting_relation_types),
                    "rationale": c.rationale,
                    "vlm_excerpt": (vlm_text or "")[:240],
                }
            )
        return cands

    def _wrap_filter_dets(self, detections, image_area):
        TRACE["yolo_raw"] = [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "area_ratio": float(d.bounding_box.area) / float(image_area)
                if image_area
                else 0.0,
            }
            for d in detections
        ]
        kept, removed = _orig_filter_dets(self, detections, image_area)
        TRACE["yolo_kept"] = [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "area_ratio": float(d.bounding_box.area) / float(image_area)
                if image_area
                else 0.0,
            }
            for d in kept
        ]
        TRACE["yolo_removed"] = [
            {"label": lab, "confidence": conf, "reason": reason}
            for lab, conf, reason in removed
        ]
        return kept, removed

    cov.ensure_salient_verified_coverage = _wrap_ensure  # type: ignore
    fact.filter_unsupported_claims_verified = _wrap_filter  # type: ignore
    bridge.extract_vlm_activity_candidates = _wrap_extract  # type: ignore
    sanity.sanitize_caption = _wrap_sanitize  # type: ignore
    yolo_mod.YoloEngine._filter_detections = _wrap_filter_dets  # type: ignore

    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from core.contracts.verified_evidence import ActivityEvidenceLevel

    print("=== STARTUP ===", flush=True)
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )

    print(f"=== ANALYZE {IMAGE} ===", flush=True)
    try:
        result = orch.analyze(PipelineRequest(image_path=IMAGE, options=opts))
    finally:
        cov.ensure_salient_verified_coverage = _orig_ensure  # type: ignore
        fact.filter_unsupported_claims_verified = _orig_filter  # type: ignore
        bridge.extract_vlm_activity_candidates = _orig_extract  # type: ignore
        sanity.sanitize_caption = _orig_sanitize  # type: ignore
        yolo_mod.YoloEngine._filter_detections = _orig_filter_dets  # type: ignore

    ctx = result.scene_context
    graph = ctx.graph
    verified = result.verified_evidence
    caption = (
        getattr(result.caption, "canonical_caption_en", None)
        or result.caption.narrative_full
        or result.caption.text
        or ""
    ).strip()
    qr = result.quality_report

    person_nodes = [n for n in graph.nodes if n.label.lower() == "person"]
    ball_nodes = [n for n in graph.nodes if n.label.lower() == "sports ball"]
    person_nodes_sorted = sorted(
        person_nodes, key=lambda n: n.bounding_box_area_ratio, reverse=True
    )

    confirmed = []
    if verified is not None:
        for a in verified.activities:
            if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe:
                confirmed.append(
                    {
                        "activity": a.activity,
                        "entity_ids": list(a.entity_ids),
                        "object_indices": list(a.object_indices),
                        "confidence": a.confidence,
                        "evidence_level": a.evidence_level.value,
                        "source": a.source,
                    }
                )

    football_acts = [
        c
        for c in confirmed
        if any(tok in c["activity"].lower() for tok in ("football", "soccer", "playing with a ball"))
    ]
    actor_person_ids: list[str] = []
    if football_acts and verified is not None:
        for eid in football_acts[0]["entity_ids"]:
            ent = verified.entity_by_id(eid)
            if ent is not None and ent.label.lower() in {
                "person",
                "man",
                "woman",
                "child",
                "people",
            }:
                actor_person_ids.append(eid)

    all_people_ids = []
    if verified is not None:
        all_people_ids = [
            e.entity_id
            for e in verified.entities
            if e.narrative_safe and e.label.lower() == "person"
        ]

    singular_fail = bool(
        re.search(
            r"\b(?:one|a)\s+person\s+is\s+playing\s+(?:football|soccer)\b",
            caption,
            re.I,
        )
    )
    has_plural_play = bool(
        re.search(
            r"\b(?:two|three)\s+people\s+are\s+playing\s+(?:football|soccer)\b"
            r"|\bpeople\s+are\s+playing\s+(?:football|soccer)\b",
            caption,
            re.I,
        )
    )
    # Global census incorrectly applied to the activity subject.
    overassigned_play = bool(
        re.search(
            r"\b(?:4|four|3|three)\s+people\s+are\s+playing\s+(?:football|soccer)\b",
            caption,
            re.I,
        )
    )
    # Prefer exact two-actor ownership when evidence has exactly 2 football actors.
    exact_two_play = bool(
        re.search(r"\btwo\s+people\s+are\s+playing\s+(?:football|soccer)\b", caption, re.I)
    )

    bridge_ok = False
    bridge_football = [b for b in TRACE["vlm_bridge"] if "football" in b["activity"].lower()]
    for b in bridge_football:
        person_idxs = [
            i
            for i, lab in zip(b["supporting_node_indices"], b["supporting_labels"])
            if lab == "person"
        ]
        if len(person_idxs) >= 2:
            bridge_ok = True
        largest = [n.index for n in person_nodes_sorted[:2]]
        if largest and set(largest).issubset(set(b["supporting_node_indices"])):
            bridge_ok = True

    two_foreground_ok = (
        len(person_nodes_sorted) >= 2
        and person_nodes_sorted[0].bounding_box_area_ratio >= 0.10
        and person_nodes_sorted[1].bounding_box_area_ratio >= 0.10
    )

    verdict = "PASS"
    reasons: list[str] = []
    if singular_fail:
        verdict = "FAIL"
        reasons.append("CRITICAL: singular 'one/a person is playing football/soccer' in final caption")
    if overassigned_play or (
        football_acts
        and len(actor_person_ids) >= 2
        and len(all_people_ids) > len(actor_person_ids)
        and re.search(
            rf"\b(?:{len(all_people_ids)}|four|three|five|six)\s+people\s+are\s+playing\b",
            caption,
            re.I,
        )
    ):
        verdict = "FAIL"
        reasons.append(
            "CRITICAL: activity subject quantity equals/uses global census "
            "(background people incorrectly assigned the activity)"
        )
    if len(actor_person_ids) >= 2 and not exact_two_play and not singular_fail:
        # Require explicit two-actor wording when evidence has exactly two actors.
        if len(actor_person_ids) == 2 and not exact_two_play:
            verdict = "FAIL"
            reasons.append(
                "shared multi-person agency not preserved as 'Two people are playing…'"
            )
    if not has_plural_play and not exact_two_play:
        if "playing" not in caption.lower():
            verdict = "FAIL"
            reasons.append("no playing activity in final caption")
    if len(actor_person_ids) < 2:
        verdict = "FAIL"
        reasons.append(f"verified football person actors < 2: {actor_person_ids}")
    if TRACE["vlm_bridge"] and bridge_football and not bridge_ok:
        verdict = "FAIL"
        reasons.append("vlm bridge did not attach both foreground people")
    if not two_foreground_ok:
        verdict = "FAIL"
        reasons.append("two large foreground person nodes not preserved")
    if (
        football_acts
        and len(all_people_ids) >= 3
        and len(actor_person_ids) >= len(all_people_ids)
    ):
        verdict = "FAIL"
        reasons.append("background people incorrectly assigned football activity")

    # Preferred / early captions from first filter before
    preferred_before = ""
    stages = TRACE["caption_stages"]
    if stages.get("filter_unsupported"):
        preferred_before = stages["filter_unsupported"][0]["before"]
    elif stages.get("sanitize"):
        preferred_before = stages["sanitize"][0]["before"]

    report = {
        "image": str(IMAGE),
        "yolo_raw_persons": sum(1 for d in TRACE["yolo_raw"] if d["label"].lower() == "person"),
        "yolo_raw_balls": sum(
            1 for d in TRACE["yolo_raw"] if d["label"].lower() == "sports ball"
        ),
        "yolo_raw": TRACE["yolo_raw"],
        "yolo_kept_persons": sum(1 for d in TRACE["yolo_kept"] if d["label"].lower() == "person"),
        "yolo_kept_balls": sum(
            1 for d in TRACE["yolo_kept"] if d["label"].lower() == "sports ball"
        ),
        "yolo_kept": TRACE["yolo_kept"],
        "yolo_removed": TRACE["yolo_removed"],
        "graph_person_count": len(person_nodes),
        "graph_ball_count": len(ball_nodes),
        "graph_person_nodes": [
            {
                "index": n.index,
                "object_id": n.object_id,
                "area_ratio": n.bounding_box_area_ratio,
                "zone": n.position_zone,
            }
            for n in person_nodes_sorted
        ],
        "graph_ball_nodes": [
            {
                "index": n.index,
                "object_id": n.object_id,
                "area_ratio": n.bounding_box_area_ratio,
            }
            for n in ball_nodes
        ],
        "vlm_bridge": TRACE["vlm_bridge"],
        "confirmed_activities": confirmed,
        "football_actor_person_ids": actor_person_ids,
        "all_narrative_people": all_people_ids,
        "people_count_verified": None if verified is None else verified.people_count,
        "preferred_before_filter": preferred_before,
        "caption_stages": TRACE["caption_stages"],
        "final_caption": caption,
        "quality": {
            "overall": None if qr is None else qr.overall_quality,
            "object_coverage": None if qr is None else qr.object_coverage,
            "activity_coverage": None if qr is None else qr.activity_coverage,
            "relationship_coverage": None if qr is None else qr.relationship_coverage,
            "evidence_consistency": None if qr is None else qr.evidence_consistency,
            "hallucination_risk": None if qr is None else qr.hallucination_risk,
        },
        "checks": {
            "singular_fail": singular_fail,
            "has_plural_play": has_plural_play,
            "bridge_ok": bridge_ok,
            "two_foreground_ok": two_foreground_ok,
            "multi_actor_ok": len(actor_person_ids) >= 2,
        },
        "verdict": verdict,
        "reasons": reasons,
        "before_broken": (
            "One person is playing football. A white sports ball sits in the scene. "
            "3 people are visible."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _print_stages(name: str) -> None:
        items = TRACE["caption_stages"].get(name, [])
        print(f"\n--- {name} ({len(items)} calls) ---", flush=True)
        for i, s in enumerate(items):
            print(f"[{i}] BEFORE:\n{s['before']}\n", flush=True)
            print(f"[{i}] AFTER:\n{s['after']}\n", flush=True)

    print("\n========== 1. YOLO ==========", flush=True)
    print(f"raw persons={report['yolo_raw_persons']} balls={report['yolo_raw_balls']}", flush=True)
    print(json.dumps(TRACE["yolo_raw"], indent=2), flush=True)
    print(f"kept persons={report['yolo_kept_persons']} balls={report['yolo_kept_balls']}", flush=True)
    print(json.dumps(TRACE["yolo_kept"], indent=2), flush=True)
    print("removed:", json.dumps(TRACE["yolo_removed"], indent=2), flush=True)

    print("\n========== 2. SCENE GRAPH ==========", flush=True)
    print(json.dumps(report["graph_person_nodes"], indent=2), flush=True)
    print(json.dumps(report["graph_ball_nodes"], indent=2), flush=True)

    print("\n========== 3. VLM ACTIVITY BRIDGE ==========", flush=True)
    print(json.dumps(TRACE["vlm_bridge"], indent=2), flush=True)

    print("\n========== 4. CONFIRMED ACTIVITIES ==========", flush=True)
    print(json.dumps(confirmed, indent=2), flush=True)
    print("football actor person ids:", actor_person_ids, flush=True)
    print("all narrative people:", all_people_ids, flush=True)

    print("\n========== 5. CAPTION STAGES ==========", flush=True)
    print("PREFERRED/BEFORE FILTER (first):", preferred_before[:500], flush=True)
    _print_stages("filter_unsupported")
    _print_stages("sanitize")
    _print_stages("ensure_salient")

    print("\n========== FINAL CAPTION ==========", flush=True)
    print(caption, flush=True)

    print("\n========== 9. QUALITY ==========", flush=True)
    print(json.dumps(report["quality"], indent=2), flush=True)

    print("\n========== COMPARE ==========", flush=True)
    print("BEFORE:", report["before_broken"], flush=True)
    print("REAL RESULT:", caption, flush=True)
    print("VERDICT:", verdict, flush=True)
    for r in reasons:
        print("-", r, flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
