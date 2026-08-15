"""Generate REAL_WORLD_EVALUATION.md from validation/real_world/results.json."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "validation" / "real_world" / "results.json"
OUT_PATH = ROOT / "docs" / "REAL_WORLD_EVALUATION.md"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_report(results: dict) -> str:
    evals = results["evaluations"]
    avg = results["averages"]
    confusion = results["confusion_summary"]

    lines = [
        "# Sentivis AI — Real-World Evaluation",
        "",
        f"**Evaluated:** {results['evaluated_at']}",
        f"**Dataset:** {results.get('dataset_source', 'COCO val2017')}",
        f"**Images:** {results['image_count']} real photographs",
        "",
        "## Methodology",
        "",
        "- Photos sourced from [COCO val2017](http://cocodataset.org/) — real-world photography, not synthetic.",
        "- Ground truth from COCO instance annotations (labels, bounding boxes, areas).",
        "- Full Sentivis pipeline executed per image (YOLO → attributes → relationships → scene graph → activities → context → BLIP → prompt → Gemma → refinement → QA).",
        "- Metrics compare pipeline output to COCO ground truth and scene-type expectations.",
        "",
        "## Average Scores",
        "",
        "| Metric | Average |",
        "|--------|---------|",
        f"| Object detection accuracy | {_pct(avg['object_detection_accuracy'])} |",
        f"| Attribute accuracy | {_pct(avg['attribute_accuracy'])} |",
        f"| Relationship correctness | {_pct(avg['relationship_correctness'])} |",
        f"| Activity reasoning | {_pct(avg['activity_reasoning'])} |",
        f"| Environment reasoning | {_pct(avg['environment_reasoning'])} |",
        f"| Caption quality | {_pct(avg['caption_quality'])} |",
        f"| Hallucination rate | {_pct(avg['hallucination_rate'])} |",
        f"| **Overall semantic score** | **{_pct(avg['overall_semantic_score'])}** |",
        "",
        "## Per-Image Comparison Table",
        "",
        "| Scene | Image | Obj Det | Attr | Rel | Activity | Env | Caption | Halluc | Overall |",
        "|-------|-------|---------|------|-----|----------|-----|---------|--------|---------|",
    ]

    for ev in evals:
        lines.append(
            f"| {ev['scene_type']} | `{ev['file_name']}` | "
            f"{ev['object_detection_accuracy']:.2f} | {ev['attribute_accuracy']:.2f} | "
            f"{ev['relationship_correctness']:.2f} | {ev['activity_reasoning']:.2f} | "
            f"{ev['environment_reasoning']:.2f} | {ev['caption_quality']:.2f} | "
            f"{ev['hallucination_rate']:.2f} | **{ev['overall_semantic_score']:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## Confusion Summary",
            "",
            f"- Total failure notes: {confusion['total_failures']}",
            f"- Images with missing important objects: {confusion['images_with_missing_objects']}/{results['image_count']}",
            f"- Images with incorrect relations: {confusion['images_with_incorrect_relations']}/{results['image_count']}",
            "",
            "### Root Cause Frequency",
            "",
        ]
    )
    for cause, count in sorted(confusion["root_cause_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"- {cause}: **{count}** images")

    # Scene-type breakdown
    lines.extend(["", "## Scores by Scene Type", "", "| Scene Type | Images | Avg Overall | Avg Obj Det | Avg Env |", "|------------|--------|-------------|-------------|---------|"])
    by_scene: dict[str, list[dict]] = {}
    for ev in evals:
        by_scene.setdefault(ev["scene_type"], []).append(ev)
    for scene, items in sorted(by_scene.items()):
        lines.append(
            f"| {scene} | {len(items)} | "
            f"{statistics.mean(i['overall_semantic_score'] for i in items):.2f} | "
            f"{statistics.mean(i['object_detection_accuracy'] for i in items):.2f} | "
            f"{statistics.mean(i['environment_reasoning'] for i in items):.2f} |"
        )

    lines.extend(["", "## Most Common Reasoning Mistakes", ""])
    mistake_counter: Counter[str] = Counter()
    for ev in evals:
        for cause in ev["root_causes"]:
            mistake_counter[cause] += 1
    for idx, (cause, count) in enumerate(mistake_counter.most_common(5), start=1):
        lines.append(f"{idx}. **{cause}** — {count} occurrences")
    if not mistake_counter:
        lines.append("- No recurring reasoning failures detected.")

    lines.extend(["", "## Per-Image Failures and Root Causes", ""])
    for ev in sorted(evals, key=lambda x: x["overall_semantic_score"]):
        if not ev["failures"]:
            continue
        lines.append(f"### `{ev['file_name']}` ({ev['scene_type']}) — score {ev['overall_semantic_score']:.2f}")
        lines.append(f"- **Caption:** {ev['caption'][:180]}{'...' if len(ev['caption']) > 180 else ''}")
        lines.append(f"- **Detected:** {', '.join(ev['detected_labels']) or 'none'}")
        lines.append(f"- **Ground truth:** {', '.join(ev['ground_truth_labels'])}")
        if ev["missing_important_objects"]:
            lines.append(f"- **Missing:** {', '.join(ev['missing_important_objects'])}")
        if ev["incorrect_relations"]:
            lines.append(f"- **Bad relations:** {', '.join(ev['incorrect_relations'])}")
        for failure in ev["failures"]:
            lines.append(f"- {failure}")
        for cause in ev["root_causes"]:
            lines.append(f"- *Root cause:* {cause}")
        lines.append("")

    lines.extend(
        [
            "## Recommendations Before Competition",
            "",
        ]
    )
    recs = _recommendations(results)
    for rec in recs:
        lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "## Heuristic Improvements Applied During Evaluation",
            "",
            "- Expanded kitchen/classroom indoor environment boosts in `context_builder.py`.",
            "- Added classroom reading activity inference when people co-occur with study objects.",
            "- Ensured `people present` is emitted whenever persons are detected alongside other activities.",
            "- Increased relationship `near_distance_ratio` from 0.18 to 0.20 for better proximity relations.",
            "- Expanded caption validator vocabulary for evidence-based context caption terms.",
            "- Down-weight umbrella-only outdoor cues when indoor/classroom/kitchen objects are present.",
            "",
            "Raw results: `validation/real_world/results.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def _recommendations(results: dict) -> list[str]:
    recs: list[str] = []
    avg = results["averages"]
    causes = results["confusion_summary"]["root_cause_counts"]
    if avg["object_detection_accuracy"] < 0.75:
        recs.append("Lower YOLO confidence threshold or use larger input size for small/occluded COCO objects.")
    if causes.get("Environment label inference gap", 0) >= 3:
        recs.append("Expand indoor/outdoor label sets for kitchen, office, and classroom COCO categories.")
    if causes.get("Relationship proximity/threshold gap", 0) >= 3:
        recs.append("Tune `near_distance_ratio` or add co-occurrence-based relation inference for distant but semantically linked objects.")
    if causes.get("Activity rule coverage gap", 0) >= 3:
        recs.append("Add scene-type-specific activity templates (classroom, kitchen, street).")
    if avg["hallucination_rate"] > 0.15:
        recs.append("Authenticate Gemma (HF_TOKEN) to reduce BLIP-only fallback hallucination risk.")
    if avg["caption_quality"] < 0.7:
        recs.append("Ensure Gemma reasoning is enabled; context-only fallbacks score lower on complex scenes.")
    recs.append("Run competition mode with deterministic seed on target hardware and cache warmed models.")
    recs.append("Review lowest-scoring images in the table above before demo — they indicate domain gaps.")
    return recs


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    OUT_PATH.write_text(generate_report(results), encoding="utf-8")
    print(f"Report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
