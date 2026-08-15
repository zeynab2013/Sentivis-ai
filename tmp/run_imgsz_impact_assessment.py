"""READ-ONLY impact assessment: YOLO imgsz=1280 (prod) vs imgsz=640 (legacy).

Does NOT modify production code. Forces imgsz only via a temporary monkeypatch
inside this process for the legacy arm.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmp.run_generalization_audit import CASES as AUDIT_CASES

FREEZE_EXTRA = [
    ("freeze_horse", "farm/horse", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("freeze_soccer", "soccer", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("freeze_motorcycle", "motorcycle", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("freeze_bicycle", "bicycle", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
]

SPORTS = frozenset(
    {
        "sports ball",
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "frisbee",
        "skis",
        "snowboard",
        "skateboard",
        "surfboard",
        "kite",
    }
)
ANIMALS = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)
SMALL_AREA = 0.02

OUT_JSON = ROOT / "tmp" / "IMGSZ_IMPACT_ASSESSMENT.json"
OUT_MD = ROOT / "tmp" / "IMGSZ_IMPACT_ASSESSMENT.md"

_LAST_DET: dict[str, Any] = {"result": None}


def _unique_images() -> list[tuple[str, str, Path]]:
    seen: set[str] = set()
    out: list[tuple[str, str, Path]] = []
    for name, cat, path in list(AUDIT_CASES) + FREEZE_EXTRA:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append((name, cat, path))
    return out


def _install_infer_patch(force_imgsz: int | None) -> None:
    """Patch YoloEngine.infer: stash DetectionResult; optionally force imgsz."""
    import time as _time
    from uuid import uuid4

    import vision.detection.yolo_engine as ye
    from core.contracts.detection import BoundingBox, Detection, DetectionResult
    from core.exceptions.vision import DetectionError

    if not hasattr(ye.YoloEngine, "_assessment_orig_infer"):
        ye.YoloEngine._assessment_orig_infer = ye.YoloEngine.infer  # type: ignore[attr-defined]

    orig = ye.YoloEngine._assessment_orig_infer  # type: ignore[attr-defined]

    def wrapped(self, image):  # type: ignore[no-untyped-def]
        if force_imgsz is None:
            result = orig(self, image)
            _LAST_DET["result"] = result
            return result

        if not self._model:
            raise DetectionError(
                "Object detection is not ready.",
                "YOLO infer called before load",
                recoverable=False,
            )
        try:
            infer_conf = min(float(self._config.confidence_threshold), ye._INFER_CONF_FLOOR)
            results = self._model.predict(
                source=image.inference_pixels,
                conf=infer_conf,
                iou=self._config.iou_threshold,
                imgsz=int(force_imgsz),
                device=self._device,
                verbose=False,
            )
        except RuntimeError as exc:
            raise DetectionError(
                "Object detection failed during analysis.",
                f"YOLO inference error: {exc}",
                recoverable=True,
            ) from exc

        raw_detections: list[Detection] = []
        source = image.source
        scale_x = source.width / image.inference_width
        scale_y = source.height / image.inference_height
        image_area = max(1.0, float(source.width * source.height))
        detected_at = _time.time()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            for box_index, box in enumerate(boxes):
                xyxy = box.xyxy.cpu().numpy().astype(float).flatten()
                bbox = BoundingBox(
                    float(xyxy[0] * scale_x),
                    float(xyxy[1] * scale_y),
                    float(xyxy[2] * scale_x),
                    float(xyxy[3] * scale_y),
                )
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = result.names.get(class_id, str(class_id))
                segmentation = self._extract_segmentation(
                    masks,
                    box_index,
                    scale_x,
                    scale_y,
                    bbox.area / image_area,
                )
                raw_detections.append(
                    Detection(
                        object_id=f"obj-{uuid4().hex[:12]}",
                        label=label,
                        confidence=confidence,
                        bounding_box=bbox,
                        class_id=class_id,
                        detected_at=detected_at,
                        segmentation=segmentation,
                    )
                )

        detections, _removed = self._filter_detections(raw_detections, image_area)
        out = DetectionResult(
            detections=tuple(detections),
            image_width=source.width,
            image_height=source.height,
            inference_timestamp=detected_at,
        )
        _LAST_DET["result"] = out
        return out

    ye.YoloEngine.infer = wrapped  # type: ignore[method-assign]


def _restore_infer() -> None:
    import vision.detection.yolo_engine as ye

    if hasattr(ye.YoloEngine, "_assessment_orig_infer"):
        ye.YoloEngine.infer = ye.YoloEngine._assessment_orig_infer  # type: ignore[attr-defined]


def _det_stats(det_result) -> dict:
    detections = list(det_result.detections)
    image_area = max(1.0, float(det_result.image_width * det_result.image_height))
    labels = Counter(d.label.lower() for d in detections)
    persons = [d for d in detections if d.label.lower() == "person"]
    sports = [d for d in detections if d.label.lower() in SPORTS]
    animals = [d for d in detections if d.label.lower() in ANIMALS]
    small = [d for d in detections if (d.bounding_box.area / image_area) < SMALL_AREA]
    primary = {"person"} | ANIMALS | {
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "airplane",
        "train",
        "boat",
    } | SPORTS
    weak = [
        d
        for d in detections
        if d.label.lower() not in primary and d.confidence < 0.50
    ]
    return {
        "n": len(detections),
        "labels": dict(labels),
        "person_n": len(persons),
        "person_confs": [round(d.confidence, 3) for d in persons],
        "person_areas": [round(d.bounding_box.area / image_area, 4) for d in persons],
        "sports_n": len(sports),
        "sports_labels": dict(Counter(d.label.lower() for d in sports)),
        "animal_n": len(animals),
        "animal_labels": dict(Counter(d.label.lower() for d in animals)),
        "small_n": len(small),
        "small_labels": dict(Counter(d.label.lower() for d in small)),
        "weak_clutter_n": len(weak),
        "weak_clutter_labels": dict(Counter(d.label.lower() for d in weak)),
    }


def _jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def _winner(delta_1280_minus_640: int, *, higher_better: bool = True) -> str:
    if delta_1280_minus_640 == 0:
        return "tie"
    if higher_better:
        return "1280" if delta_1280_minus_640 > 0 else "640"
    return "1280" if delta_1280_minus_640 < 0 else "640"


def _quality_overall(qr) -> float | None:
    if qr is None:
        return None
    for attr in ("overall_score", "overall", "score"):
        if hasattr(qr, attr):
            val = getattr(qr, attr)
            if val is not None:
                return float(val)
    return None


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from core.contracts.verified_evidence import ActivityEvidenceLevel

    images = _unique_images()
    print(f"ASSESSMENT images={len(images)}", flush=True)
    for n, c, p in images:
        print(f"  - {n} | {c} | {p.name}", flush=True)

    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )

    rows: list[dict] = []
    try:
        for name, cat, path in images:
            print(f"\n=== {name} {path.name} ===", flush=True)
            arms: dict[str, dict] = {}
            for arm, force in (("1280", None), ("640", 640)):
                _install_infer_patch(force)
                _LAST_DET["result"] = None
                t0 = time.time()
                result = orch.analyze(PipelineRequest(image_path=path, options=opts))
                elapsed = time.time() - t0
                ctx = result.scene_context
                ve = result.verified_evidence
                det_result = _LAST_DET["result"]
                if det_result is None:
                    raise RuntimeError(f"No DetectionResult stashed for {arm} on {path.name}")
                dstat = _det_stats(det_result)
                graph_labels = Counter(n.label.lower() for n in ctx.graph.nodes)
                confirmed = []
                if ve is not None:
                    for a in ve.activities:
                        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED:
                            confirmed.append(
                                {
                                    "activity": a.activity,
                                    "actors": list(a.entity_ids),
                                    "conf": round(a.confidence, 3),
                                }
                            )
                caption = (
                    getattr(result.caption, "canonical_caption_en", None)
                    or getattr(result.caption, "narrative_full", None)
                    or result.caption.text
                    or ""
                ).strip()
                arms[arm] = {
                    "elapsed_s": round(elapsed, 1),
                    "detection": dstat,
                    "graph_labels": dict(graph_labels),
                    "graph_n": len(ctx.graph.nodes),
                    "people_count": None if ve is None else ve.people_count,
                    "confirmed_activities": confirmed,
                    "caption": caption,
                    "quality_overall": _quality_overall(result.quality_report),
                    "qa_passed": bool(result.qa_passed),
                }
                print(
                    f"  [{arm}] persons={dstat['person_n']} sports={dstat['sports_n']} "
                    f"animals={dstat['animal_n']} small={dstat['small_n']} "
                    f"weak={dstat['weak_clutter_n']} graph={len(ctx.graph.nodes)} "
                    f"acts={len(confirmed)} q={arms[arm]['quality_overall']}",
                    flush=True,
                )

            a, b = arms["1280"], arms["640"]
            da, db = a["detection"], b["detection"]
            g_jac = _jaccard(Counter(a["graph_labels"]), Counter(b["graph_labels"]))
            person_winner = _winner(da["person_n"] - db["person_n"])
            sports_winner = _winner(da["sports_n"] - db["sports_n"])
            animal_winner = _winner(da["animal_n"] - db["animal_n"])
            small_winner = _winner(da["small_n"] - db["small_n"])
            fp_winner = _winner(da["weak_clutter_n"] - db["weak_clutter_n"], higher_better=False)
            graph_same = a["graph_labels"] == b["graph_labels"]
            act_1280 = {x["activity"] for x in a["confirmed_activities"]}
            act_640 = {x["activity"] for x in b["confirmed_activities"]}
            cap_same = a["caption"].strip().lower() == b["caption"].strip().lower()

            votes: Counter[str] = Counter()
            for w in (person_winner, sports_winner, animal_winner, small_winner, fp_winner):
                if w in ("1280", "640"):
                    votes[w] += 1
            if votes["1280"] > votes["640"]:
                pref = "1280"
            elif votes["640"] > votes["1280"]:
                pref = "640"
            else:
                pref = "equivalent"

            flags = []
            if da["person_n"] != db["person_n"]:
                flags.append(f"person_delta={da['person_n']}-{db['person_n']}")
            if da["sports_n"] != db["sports_n"]:
                flags.append(f"sports_delta={da['sports_n']}-{db['sports_n']}")
            if da["animal_n"] != db["animal_n"]:
                flags.append(f"animal_delta={da['animal_n']}-{db['animal_n']}")
            if not graph_same:
                flags.append("graph_diff")
            if act_1280 != act_640:
                flags.append("activity_diff")
            if not cap_same:
                flags.append("caption_diff")

            rows.append(
                {
                    "id": name,
                    "category": cat,
                    "path": str(path),
                    "file": path.name,
                    "arms": arms,
                    "deltas": {
                        "person": da["person_n"] - db["person_n"],
                        "sports": da["sports_n"] - db["sports_n"],
                        "animal": da["animal_n"] - db["animal_n"],
                        "small": da["small_n"] - db["small_n"],
                        "weak_clutter": da["weak_clutter_n"] - db["weak_clutter_n"],
                        "graph_jaccard": round(g_jac, 3),
                        "graph_identical": graph_same,
                        "activity_identical": act_1280 == act_640,
                        "caption_identical": cap_same,
                    },
                    "winners": {
                        "person": person_winner,
                        "sports": sports_winner,
                        "animal": animal_winner,
                        "small": small_winner,
                        "fp_pressure": fp_winner,
                    },
                    "preference": pref,
                    "flags": flags,
                }
            )
            # checkpoint after each image
            OUT_JSON.write_text(
                json.dumps({"partial": True, "rows_so_far": rows}, indent=2),
                encoding="utf-8",
            )
    finally:
        _restore_infer()

    better_1280 = [r for r in rows if r["preference"] == "1280"]
    better_640 = [r for r in rows if r["preference"] == "640"]
    equiv = [r for r in rows if r["preference"] == "equivalent"]

    def tally(key: str) -> dict:
        return dict(Counter(r["winners"][key] for r in rows))

    summary = {
        "n_images": len(rows),
        "preference_counts": {
            "1280": len(better_1280),
            "640": len(better_640),
            "equivalent": len(equiv),
        },
        "metric_winners": {
            "person": tally("person"),
            "sports": tally("sports"),
            "animal": tally("animal"),
            "small": tally("small"),
            "fp_pressure": tally("fp_pressure"),
        },
        "person_regressions_at_1280": [r["file"] for r in rows if r["deltas"]["person"] < 0],
        "sports_regressions_at_1280": [r["file"] for r in rows if r["deltas"]["sports"] < 0],
        "animal_regressions_at_1280": [r["file"] for r in rows if r["deltas"]["animal"] < 0],
        "person_gains_at_1280": [r["file"] for r in rows if r["deltas"]["person"] > 0],
        "sports_gains_at_1280": [r["file"] for r in rows if r["deltas"]["sports"] > 0],
        "animal_gains_at_1280": [r["file"] for r in rows if r["deltas"]["animal"] > 0],
        "images_1280_better": [r["file"] for r in better_1280],
        "images_640_better": [r["file"] for r in better_640],
        "images_equivalent": [r["file"] for r in equiv],
        "graph_identical_count": sum(1 for r in rows if r["deltas"]["graph_identical"]),
        "activity_identical_count": sum(1 for r in rows if r["deltas"]["activity_identical"]),
        "caption_identical_count": sum(1 for r in rows if r["deltas"]["caption_identical"]),
    }

    person_loss = len(summary["person_regressions_at_1280"])
    person_gain = len(summary["person_gains_at_1280"])
    sports_loss = len(summary["sports_regressions_at_1280"])
    sports_gain = len(summary["sports_gains_at_1280"])
    pref_1280 = summary["preference_counts"]["1280"]
    pref_640 = summary["preference_counts"]["640"]

    # Conservative recommendation policy (user: insufficient evidence → A)
    if len(rows) < 5:
        rec, rec_text = "A", "Keep 1280 for final freeze"
        rationale = "Evidence set too small for a verified change; defaulting to A."
    elif sports_loss == 0 and person_loss <= 1 and pref_1280 >= pref_640:
        rec, rec_text = "A", "Keep 1280 for final freeze"
        rationale = (
            "1280 is equal or better on most images; sports recall does not regress vs 640; "
            "the known farm second-person miss is a single-image failure, not a distribution-wide "
            "collapse. Reverting globally would be an unverified change relative to the freeze path."
        )
    elif pref_640 > pref_1280 + 1 and (person_loss + sports_loss) > (person_gain + sports_gain + 1):
        rec, rec_text = "B", "Revert to 640"
        rationale = (
            "Legacy 640 wins on a clear majority of validation images with material "
            "person/sports recall advantages and without offsetting 1280 gains."
        )
    else:
        rec = "C"
        rec_text = (
            "Multi-scale detection is justified as a future improvement, "
            "but keep 1280 for final freeze"
        )
        rationale = (
            "Tradeoffs are mixed across the validation distribution (e.g. occasional person "
            "recall at 640 vs sports/small-object advantages at 1280). A global revert is not "
            "clearly safer; multi-scale is the correct future fix while freezing on 1280."
        )

    payload = {
        "summary": summary,
        "recommendation": {"code": rec, "text": rec_text, "rationale": rationale},
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IMGSZ IMPACT ASSESSMENT (read-only)",
        "",
        f"Images compared: **{len(rows)}** unique real validation images",
        "(generalization audit set ∪ competition freeze critical set).",
        "",
        "Production code was **not** modified. Legacy arm forces `imgsz=640` only inside this harness.",
        "",
        "## Recommendation",
        "",
        f"**{rec}) {rec_text}**",
        "",
        rationale,
        "",
        "## Preference counts",
        "",
        f"- 1280 better: {len(better_1280)} — {', '.join(r['file'] for r in better_1280) or '(none)'}",
        f"- 640 better: {len(better_640)} — {', '.join(r['file'] for r in better_640) or '(none)'}",
        f"- Equivalent: {len(equiv)} — {', '.join(r['file'] for r in equiv) or '(none)'}",
        "",
        "## Metric winner tallies (per image)",
        "",
        f"- person recall: {summary['metric_winners']['person']}",
        f"- sports-object recall: {summary['metric_winners']['sports']}",
        f"- animal recall: {summary['metric_winners']['animal']}",
        f"- small-object recall: {summary['metric_winners']['small']}",
        f"- false-positive pressure (fewer weak clutter better): {summary['metric_winners']['fp_pressure']}",
        "",
        "## Downstream stability",
        "",
        f"- identical scene graphs: {summary['graph_identical_count']}/{len(rows)}",
        f"- identical confirmed activities: {summary['activity_identical_count']}/{len(rows)}",
        f"- identical captions: {summary['caption_identical_count']}/{len(rows)}",
        "",
        "## Important regressions @1280 vs @640",
        "",
        f"- person losses: {summary['person_regressions_at_1280'] or 'none'}",
        f"- person gains: {summary['person_gains_at_1280'] or 'none'}",
        f"- sports losses: {summary['sports_regressions_at_1280'] or 'none'}",
        f"- sports gains: {summary['sports_gains_at_1280'] or 'none'}",
        f"- animal losses: {summary['animal_regressions_at_1280'] or 'none'}",
        f"- animal gains: {summary['animal_gains_at_1280'] or 'none'}",
        "",
        "## Per-image detail",
        "",
    ]
    for r in rows:
        d = r["deltas"]
        a = r["arms"]["1280"]
        b = r["arms"]["640"]
        lines += [
            f"### {r['file']} (`{r['id']}` / {r['category']})",
            f"- preference: **{r['preference']}** flags={r['flags'] or []}",
            (
                f"- persons 1280/640: {a['detection']['person_n']}/{b['detection']['person_n']} "
                f"(confs {a['detection']['person_confs']} vs {b['detection']['person_confs']})"
            ),
            (
                f"- sports {a['detection']['sports_n']}/{b['detection']['sports_n']} "
                f"{a['detection']['sports_labels']} vs {b['detection']['sports_labels']}"
            ),
            (
                f"- animals {a['detection']['animal_n']}/{b['detection']['animal_n']} "
                f"{a['detection']['animal_labels']} vs {b['detection']['animal_labels']}"
            ),
            (
                f"- small {a['detection']['small_n']}/{b['detection']['small_n']}; "
                f"weak_clutter {a['detection']['weak_clutter_n']}/{b['detection']['weak_clutter_n']}"
            ),
            (
                f"- graph_n {a['graph_n']}/{b['graph_n']} jaccard={d['graph_jaccard']} "
                f"identical={d['graph_identical']}"
            ),
            f"- confirmed activities 1280: {[x['activity'] for x in a['confirmed_activities']]}",
            f"- confirmed activities 640: {[x['activity'] for x in b['confirmed_activities']]}",
            f"- quality/qa 1280: {a['quality_overall']}/{a['qa_passed']} | 640: {b['quality_overall']}/{b['qa_passed']}",
            f"- caption 1280: {a['caption'][:220]}",
            f"- caption 640: {b['caption'][:220]}",
            "",
        ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"RECOMMENDATION: {rec}) {rec_text}", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
