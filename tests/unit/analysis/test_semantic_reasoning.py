"""Unit tests for semantic synthesis response parsing."""

import json

from analysis.semantic.semantic_response_parser import parse_semantic_response


def test_parse_semantic_json() -> None:
    payload = {
        "scene_explanation": "A person stands near a tennis racket outdoors.",
        "rejected_conclusions": ["swimming"],
        "contradictions_resolved": [],
        "caption": "A person appears near a tennis racket in an outdoor setting.",
    }
    parsed = parse_semantic_response(json.dumps(payload))
    assert "tennis racket" in parsed.caption
    assert parsed.rejected_conclusions == ("swimming",)
