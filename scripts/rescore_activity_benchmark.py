"""Rescore benchmark JSON with updated activity synonym matching."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from core.contracts.analysis import ActivityEvidence, ActivityHints

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_real_world_evaluation import _score_activities

PATH = ROOT / "validation" / "real_world" / "activity_benchmark_comparison.json"


def _activities_from_caption(caption: str) -> ActivityHints:
    lower = caption.lower()
    activities: list[str] = []
    match = re.search(r"supported activity:\s*([^(.\n]+)", lower)
    if match:
        activities.append(match.group(1).strip())
    for label in ("people present", "playing sports", "playing tennis", "dining", "working", "reading"):
        if label in lower and label not in activities:
            activities.append(label)
    if not activities:
        activities.append("static scene")
    return ActivityHints(
        activities=tuple(
            ActivityEvidence(activity, 0.7, (), (), "") for activity in dict.fromkeys(activities)
        ),
        confidence=0.7,
    )


def rescore() -> dict:
    data = json.loads(PATH.read_text(encoding="utf-8"))

    def process(evals: list[dict]) -> None:
        for ev in evals:
            hints = _activities_from_caption(ev.get("caption", ""))
            score, _ = _score_activities(hints, set(ev["ground_truth_labels"]), ev["scene_type"])
            ev["activity_reasoning"] = round(score, 3)
            parts = [
                ev["object_detection_accuracy"],
                ev["attribute_accuracy"],
                ev["relationship_correctness"],
                ev["activity_reasoning"],
                ev["environment_reasoning"],
                ev["caption_quality"],
                1.0 - ev["hallucination_rate"],
            ]
            ev["overall_semantic_score"] = round(sum(parts) / len(parts), 3)

    process(data["heuristic_evaluations"])
    if data.get("ollama_evaluations"):
        process(data["ollama_evaluations"])

    def summary(evals: list[dict], mode: str, duration: float) -> dict:
        return {
            "mode": mode,
            "activity_reasoning": round(statistics.mean(e["activity_reasoning"] for e in evals), 3),
            "relationship_correctness": round(
                statistics.mean(e["relationship_correctness"] for e in evals), 3
            ),
            "caption_quality": round(statistics.mean(e["caption_quality"] for e in evals), 3),
            "hallucination_rate": round(statistics.mean(e["hallucination_rate"] for e in evals), 3),
            "overall_semantic_score": round(statistics.mean(e["overall_semantic_score"] for e in evals), 3),
            "duration_seconds": duration,
        }

    data["heuristic"] = summary(
        data["heuristic_evaluations"], "heuristic", data["heuristic"]["duration_seconds"]
    )
    if data.get("ollama_evaluations"):
        data["ollama"] = summary(
            data["ollama_evaluations"], "ollama", data["ollama"]["duration_seconds"]
        )
        h, o = data["heuristic"], data["ollama"]
        h_score = (
            h["activity_reasoning"] * 0.35
            + h["caption_quality"] * 0.25
            + (1 - h["hallucination_rate"]) * 0.2
            + h["relationship_correctness"] * 0.2
        )
        o_score = (
            o["activity_reasoning"] * 0.35
            + o["caption_quality"] * 0.25
            + (1 - o["hallucination_rate"]) * 0.2
            + o["relationship_correctness"] * 0.2
        )
        data["winner"] = "ollama" if o_score >= h_score else "heuristic"

    PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    result = rescore()
    print("heuristic:", result["heuristic"])
    print("ollama:", result.get("ollama"))
    print("winner:", result.get("winner"))
