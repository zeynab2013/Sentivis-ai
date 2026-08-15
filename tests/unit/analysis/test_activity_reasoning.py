"""Unit tests for Ollama activity response parsing."""

import json

from analysis.activity.activity_response_parser import parse_activity_response
from core.contracts.analysis import SceneGraph, SceneNode


def _graph() -> SceneGraph:
    return SceneGraph(
        nodes=(
            SceneNode(0, "obj-0", "person", 0.05, "middle-center"),
            SceneNode(1, "obj-1", "tennis racket", 0.02, "middle-left"),
        ),
        relations=(),
    )


def test_parse_valid_activity_json() -> None:
    payload = {
        "activity": "playing sports",
        "confidence": 0.82,
        "evidence": ["person near tennis racket"],
        "rejected_conclusions": ["swimming"],
        "supporting_object_indices": [0, 1],
        "supporting_relation_types": ["near"],
        "caption": "A person appears to be playing tennis.",
    }
    parsed = parse_activity_response(json.dumps(payload), _graph(), empty_confidence=0.4)
    assert parsed.activities.activities[0].activity == "playing sports"
    assert parsed.caption.startswith("A person")
    assert "swimming" in parsed.rejected_conclusions
