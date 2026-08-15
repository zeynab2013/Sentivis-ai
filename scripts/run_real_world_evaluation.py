"""Run Sentivis pipeline on real-world validation photos and score against COCO ground truth."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from app.container import DependencyContainer  # noqa: E402
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config  # noqa: E402
from core.contracts.pipeline import AnalysisOptions, PipelineRequest  # noqa: E402
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator  # noqa: E402

DATASET_DIR = ROOT / "validation" / "real_world"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
RESULTS_PATH = DATASET_DIR / "results.json"

_OUTDOOR_GT = {
    "tree", "car", "bus", "truck", "motorcycle", "bicycle", "bench", "bird", "dog",
    "fire hydrant", "stop sign", "parking meter", "traffic light", "umbrella", "kite",
    "skateboard", "surfboard", "sports ball", "frisbee",
}
_INDOOR_GT = {
    "chair", "couch", "bed", "tv", "laptop", "dining table", "refrigerator", "microwave",
    "oven", "sink", "toaster", "keyboard", "mouse", "book", "clock", "cell phone",
}
_NEVER_CONTAINERS = {
    "sports ball", "tennis racket", "person", "people", "man", "woman", "child",
    "cell phone", "book", "bottle", "bird", "cat", "dog",
}
_PERSON = {"person"}
_SPORT_ITEMS = {"sports ball", "tennis racket", "skateboard", "surfboard", "kite", "baseball bat", "frisbee"}
_VEHICLE = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}
_FOOD = {"dining table", "pizza", "bowl", "sandwich", "oven", "refrigerator", "sink", "microwave", "cup", "bottle"}
_TECH = {"laptop", "keyboard", "mouse", "cell phone", "tv", "book"}
_ANIMALS = {"dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear"}


@dataclass
class ImageEvaluation:
    scene_type: str
    file_name: str
    image_path: str
    duration_seconds: float
    caption: str
    detected_labels: list[str]
    ground_truth_labels: list[str]
    object_detection_accuracy: float
    attribute_accuracy: float
    relationship_correctness: float
    activity_reasoning: float
    environment_reasoning: float
    caption_quality: float
    hallucination_rate: float
    evidence_consistency: float
    narrative_fluency: float
    narrative_full: str
    narrative_short: str
    missing_important_objects: list[str]
    incorrect_relations: list[str]
    overall_semantic_score: float
    failures: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)


def _coco_bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _expected_zone(cx: float, cy: float, image_w: int, image_h: int) -> str:
    nx = cx / max(image_w, 1)
    ny = cy / max(image_h, 1)
    horizontal = "left" if nx < 0.33 else "center" if nx < 0.66 else "right"
    vertical = "top" if ny < 0.33 else "middle" if ny < 0.66 else "bottom"
    return f"{vertical}-{horizontal}"


def _important_gt_objects(gt_objects: list[dict], image_area: float) -> list[dict]:
    return [
        obj
        for obj in gt_objects
        if obj["area"] / max(image_area, 1.0) >= 0.005
        or obj["label"] in _PERSON | _VEHICLE | _ANIMALS | _SPORT_ITEMS
    ]


def _expected_environment(gt_labels: set[str], scene_type: str) -> str:
    outdoor = len(gt_labels & _OUTDOOR_GT)
    indoor = len(gt_labels & _INDOOR_GT)
    if scene_type in {"kitchens", "offices", "classrooms", "indoor"}:
        indoor += 2
    if scene_type in {"streets", "vehicles", "outdoor", "sports"}:
        outdoor += 1
    if outdoor > indoor:
        return "outdoor"
    if indoor > 0:
        return "indoor"
    return "unknown"


def _expected_activities(gt_labels: set[str], scene_type: str) -> set[str]:
    expected: set[str] = set()
    if gt_labels & _PERSON:
        expected.add("people present")
    if gt_labels & _PERSON and gt_labels & _SPORT_ITEMS:
        expected.add("playing sports")
    if gt_labels & _PERSON and gt_labels & _FOOD:
        expected.add("dining")
    if gt_labels & _PERSON and gt_labels & _TECH:
        expected.add("working")
    if gt_labels & _PERSON and gt_labels & {"book"}:
        expected.add("reading")
    if gt_labels & _VEHICLE:
        expected.add("transportation scene")
    if scene_type == "classrooms" and gt_labels & _PERSON:
        expected.add("people present")
    if scene_type == "animals" and gt_labels & _ANIMALS and not gt_labels & _PERSON:
        expected.add("static scene")
    if not expected:
        expected.add("static scene")
    return expected


def _expected_relations(gt_objects: list[dict], image_w: int, image_h: int) -> set[tuple[str, str, str]]:
    expected: set[tuple[str, str, str]] = set()
    diagonal = (image_w**2 + image_h**2) ** 0.5
    for i, a in enumerate(gt_objects):
        for j, b in enumerate(gt_objects):
            if i == j:
                continue
            cx_a, cy_a = _coco_bbox_center(a["bbox"])
            cx_b, cy_b = _coco_bbox_center(b["bbox"])
            dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
            la, lb = a["label"].lower(), b["label"].lower()
            if dist > diagonal * 0.22:
                continue
            if la in _PERSON and lb in _SPORT_ITEMS:
                expected.add((la, "playing_with", lb))
            elif la in _PERSON and lb in _TECH:
                expected.add((la, "holding", lb))
            elif la in _PERSON and lb in _PERSON:
                expected.add((la, "standing_beside", lb))
            elif la in _VEHICLE or lb in _VEHICLE:
                expected.add((la, "near_vehicle", lb))
    return expected


def _score_object_detection(
    gt_objects: list[dict],
    detected_labels: list[str],
    image_area: float,
) -> tuple[float, list[str], list[str]]:
    important = _important_gt_objects(gt_objects, image_area)
    if not important:
        important = gt_objects
    gt_counts = Counter(obj["label"] for obj in important)
    det_counts = Counter(detected_labels)
    matched = sum(min(det_counts[label], count) for label, count in gt_counts.items())
    total_gt = sum(gt_counts.values())
    total_det = sum(det_counts.values())
    recall = matched / max(total_gt, 1)
    precision = matched / max(total_det, 1) if total_det else (1.0 if total_gt == 0 else 0.0)
    accuracy = (recall + precision) / 2.0
    missing = sorted(label for label, count in gt_counts.items() if det_counts[label] < count)
    failures: list[str] = []
    if missing:
        failures.append(f"Missed GT objects: {', '.join(missing)}")
    extra_counts = Counter(detected_labels)
    for label, count in gt_counts.items():
        extra_counts[label] -= min(extra_counts[label], count)
    extra = sorted(label for label, count in extra_counts.items() if count > 0 for _ in range(count))
    if extra:
        failures.append(f"Extra detections: {', '.join(extra)}")
    return accuracy, missing, failures


def _score_attributes(
    gt_objects: list[dict],
    graph,
    attributes,
    image_w: int,
    image_h: int,
) -> tuple[float, list[str]]:
    if not gt_objects:
        return 1.0, []
    attrs_by_index: dict[int, dict[str, str]] = {}
    for attr in attributes.attributes:
        attrs_by_index.setdefault(attr.object_index, {})[attr.name] = attr.value
    nodes_by_label: dict[str, list] = {}
    for node in graph.nodes:
        nodes_by_label.setdefault(node.label.lower(), []).append(node)

    checks = 0
    passed = 0
    failures: list[str] = []
    for gt in _important_gt_objects(gt_objects, float(image_w * image_h)) or gt_objects:
        label = gt["label"].lower()
        if label not in nodes_by_label or not nodes_by_label[label]:
            continue
        node = nodes_by_label[label].pop(0)
        checks += 3
        det_attrs = attrs_by_index.get(node.index, {})
        cx, cy = _coco_bbox_center(gt["bbox"])
        expected_zone = _expected_zone(cx, cy, image_w, image_h)
        actual_zone = det_attrs.get("position_zone") or node.position_zone
        if actual_zone == expected_zone or (
            expected_zone.split("-")[0] == actual_zone.split("-")[0]
        ):
            passed += 1
        else:
            failures.append(f"Zone mismatch for {label}: expected {expected_zone}, got {actual_zone}")
        if det_attrs.get("relative_size"):
            passed += 1
        if det_attrs.get("color"):
            passed += 1
        else:
            failures.append(f"Missing color attribute for {label}")
    return (passed / checks if checks else 1.0), failures


def _find_incorrect_relations(graph) -> list[str]:
    labels = {node.index: node.label.lower() for node in graph.nodes}
    issues: list[str] = []
    for rel in graph.relations:
        if rel.relation_type != "inside":
            continue
        inner = labels.get(rel.subject_index, "")
        outer = labels.get(rel.object_index, "")
        if inner in _PERSON and outer in _NEVER_CONTAINERS:
            issues.append(f"{inner} inside {outer}")
        if inner in _PERSON and outer in _PERSON:
            issues.append(f"{inner} inside {outer}")
    return issues


def _score_relationships(graph, gt_objects: list[dict], image_w: int, image_h: int) -> tuple[float, list[str], list[str]]:
    incorrect = _find_incorrect_relations(graph)
    expected = _expected_relations(gt_objects, image_w, image_h)
    if not expected:
        return (1.0 if not incorrect else max(0.0, 1.0 - 0.25 * len(incorrect))), incorrect, []
    nodes = {node.index: node.label.lower() for node in graph.nodes}
    found = 0
    failures: list[str] = []
    for subj_label, rel_type, obj_label in expected:
        ok = any(
            rel.relation_type == rel_type
            and nodes.get(rel.subject_index) == subj_label
            and nodes.get(rel.object_index) == obj_label
            for rel in graph.relations
        )
        if ok:
            found += 1
        else:
            near_ok = any(
                rel.relation_type in {rel_type, "near", "playing_with", "holding", "near_vehicle"}
                and nodes.get(rel.subject_index) == subj_label
                and nodes.get(rel.object_index) == obj_label
                for rel in graph.relations
            )
            if near_ok:
                found += 0.5
            else:
                failures.append(f"Missing expected relation: {subj_label} {rel_type} {obj_label}")
    base = found / len(expected)
    penalty = min(0.5, 0.15 * len(incorrect))
    return max(0.0, base - penalty), incorrect, failures


_ACTIVITY_EQUIVALENTS: dict[str, set[str]] = {
    "people present": {"people present", "person present", "standing", "waiting", "walking"},
    "playing sports": {"playing sports", "playing tennis", "playing baseball", "playing soccer", "playing", "tennis"},
    "dining": {"dining", "eating", "meal"},
    "working": {"working", "work", "using computer", "typing"},
    "reading": {"reading", "studying", "read"},
    "transportation scene": {"transportation scene", "driving", "travel", "commuting"},
    "static scene": {"static scene", "idle", "still"},
}


def _activity_matches(predicted: str, expected: str) -> bool:
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()
    if predicted == expected:
        return True
    equivalents = _ACTIVITY_EQUIVALENTS.get(expected, set())
    if predicted in equivalents:
        return True
    return any(token in predicted for token in expected.split()) or any(token in expected for token in predicted.split())


def _score_activities(activities, gt_labels: set[str], scene_type: str) -> tuple[float, list[str]]:
    expected = _expected_activities(gt_labels, scene_type)
    predicted = {item.activity.lower() for item in activities.activities}
    if not expected:
        return 1.0, []
    matched = 0
    for exp in expected:
        if any(_activity_matches(pred, exp) for pred in predicted):
            matched += 1
    score = matched / max(len(expected), 1)
    failures = []
    if score < 1.0:
        failures.append(f"Expected activities {sorted(expected)}, got {sorted(predicted)}")
    return score, failures


def _score_environment(env, gt_labels: set[str], scene_type: str) -> tuple[float, list[str]]:
    expected = _expected_environment(gt_labels, scene_type)
    if expected == "unknown":
        return (0.7 if env.indoor_outdoor == "unknown" else 0.5), []
    if env.indoor_outdoor == expected:
        return 1.0, []
    if env.indoor_outdoor == "unknown":
        return 0.5, [f"Environment unknown; GT/scene suggests {expected}"]
    return 0.0, [f"Environment {env.indoor_outdoor} contradicts GT/scene {expected}"]


def _aggregate_failures(*parts: list[str]) -> tuple[list[str], list[str]]:
    failures = [f for part in parts for f in part if f]
    causes: list[str] = []
    for f in failures:
        lower = f.lower()
        if "missed gt" in lower:
            causes.append("Detection recall gap (YOLO threshold or object scale)")
        elif "extra detection" in lower:
            causes.append("Detection precision gap (false positive)")
        elif "inside" in lower:
            causes.append("Invalid containment heuristic")
        elif "missing expected relation" in lower:
            causes.append("Relationship proximity/threshold gap")
        elif "environment" in lower:
            causes.append("Environment label inference gap")
        elif "expected activities" in lower:
            causes.append("Activity rule coverage gap")
        elif "zone mismatch" in lower:
            causes.append("Attribute zone boundary edge case")
    return failures, sorted(set(causes))


def _gt_hallucination_rate(detected_labels: list[str], gt_labels: list[str], caption: str, context) -> float:
    """Estimate hallucination from extra detections and unsupported caption object tokens."""
    from language.validation.caption_validator import CaptionEvidenceValidator

    gt_set = {label.lower() for label in gt_labels}
    extra_det = [label for label in detected_labels if label.lower() not in gt_set]
    det_rate = len(extra_det) / max(len(detected_labels), 1)
    unsupported = CaptionEvidenceValidator().unsupported_object_tokens(caption, context)
    cap_rate = min(1.0, len(unsupported) * 0.25)
    if not detected_labels and not unsupported:
        return 0.0
    return round((det_rate + cap_rate) / 2.0, 3)


def evaluate_image(orchestrator, item: dict) -> ImageEvaluation:
    image_path = ROOT / item["local_path"]
    gt_objects = item["ground_truth_objects"]
    gt_labels = set(item["ground_truth_labels"])
    scene_type = item["scene_type"]
    image_w, image_h = item["width"], item["height"]
    image_area = float(image_w * image_h)

    started = time.perf_counter()
    result = orchestrator.analyze(PipelineRequest(image_path, AnalysisOptions(enable_gemma=True)))
    duration = time.perf_counter() - started

    graph = result.scene_context.graph
    detected_labels = [node.label for node in graph.nodes]

    obj_acc, missing, obj_failures = _score_object_detection(gt_objects, detected_labels, image_area)
    attr_acc, attr_failures = _score_attributes(
        gt_objects, graph, result.scene_context.attributes, image_w, image_h
    )
    rel_score, incorrect_rels, rel_failures = _score_relationships(graph, gt_objects, image_w, image_h)
    act_score, act_failures = _score_activities(result.scene_context.activities, gt_labels, scene_type)
    env_score, env_failures = _score_environment(result.scene_context.environment, gt_labels, scene_type)

    narrative_full = result.caption.narrative_full.strip() or result.caption.text
    narrative_short = result.caption.narrative_short.strip() or result.caption.text
    narrative_quality = CaptionQualityEvaluator().evaluate(narrative_full, result.scene_context)
    caption_quality = narrative_quality.overall_quality
    evidence_consistency = narrative_quality.evidence_consistency
    narrative_fluency = narrative_quality.fluency_score
    hallucination_rate = _gt_hallucination_rate(
        detected_labels, sorted(gt_labels), narrative_full, result.scene_context
    )

    scores = [obj_acc, attr_acc, rel_score, act_score, env_score, caption_quality, 1.0 - hallucination_rate]
    overall = sum(scores) / len(scores)

    failures, root_causes = _aggregate_failures(
        obj_failures, attr_failures, rel_failures, act_failures, env_failures
    )

    return ImageEvaluation(
        scene_type=scene_type,
        file_name=item["file_name"],
        image_path=str(image_path),
        duration_seconds=round(duration, 2),
        caption=result.caption.text,
        narrative_full=narrative_full,
        narrative_short=narrative_short,
        detected_labels=sorted(set(detected_labels)),
        ground_truth_labels=sorted(gt_labels),
        object_detection_accuracy=round(obj_acc, 3),
        attribute_accuracy=round(attr_acc, 3),
        relationship_correctness=round(rel_score, 3),
        activity_reasoning=round(act_score, 3),
        environment_reasoning=round(env_score, 3),
        caption_quality=round(caption_quality, 3),
        hallucination_rate=round(hallucination_rate, 3),
        evidence_consistency=round(evidence_consistency, 3),
        narrative_fluency=round(narrative_fluency, 3),
        missing_important_objects=missing,
        incorrect_relations=incorrect_rels,
        overall_semantic_score=round(overall, 3),
        failures=failures,
        root_causes=root_causes,
    )


def run_evaluation() -> dict:
    if not MANIFEST_PATH.exists():
        from scripts.build_real_world_dataset import build_dataset

        build_dataset()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ctx = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    orchestrator = ctx.main_controller.pipeline._orchestrator  # noqa: SLF001

    evaluations: list[ImageEvaluation] = []
    for idx, item in enumerate(manifest["images"], start=1):
        print(f"[{idx}/{len(manifest['images'])}] {item['scene_type']}: {item['file_name']}", flush=True)
        evaluations.append(evaluate_image(orchestrator, item))

    ctx.model_manager.release_all()
    ctx.memory_manager.clear_gpu_cache()

    numeric_keys = [
        "object_detection_accuracy",
        "attribute_accuracy",
        "relationship_correctness",
        "activity_reasoning",
        "environment_reasoning",
        "caption_quality",
        "hallucination_rate",
        "evidence_consistency",
        "narrative_fluency",
        "overall_semantic_score",
    ]
    averages = {key: round(statistics.mean(getattr(e, key) for e in evaluations), 3) for key in numeric_keys}

    all_causes: dict[str, int] = {}
    for ev in evaluations:
        for cause in ev.root_causes:
            all_causes[cause] = all_causes.get(cause, 0) + 1

    payload = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "image_count": len(evaluations),
        "dataset_source": manifest.get("source"),
        "averages": averages,
        "confusion_summary": {
            "total_failures": sum(len(e.failures) for e in evaluations),
            "images_with_missing_objects": sum(1 for e in evaluations if e.missing_important_objects),
            "images_with_incorrect_relations": sum(1 for e in evaluations if e.incorrect_relations),
            "root_cause_counts": all_causes,
        },
        "evaluations": [asdict(e) for e in evaluations],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    run_evaluation()
