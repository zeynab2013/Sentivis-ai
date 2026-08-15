"""Parse and validate Ollama activity reasoning JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.contracts.analysis import ActivityEvidence, ActivityHints, SceneGraph
from core.logging import get_logger

logger = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedActivityReasoning:
    """Validated LLM activity output."""

    activities: ActivityHints
    caption: str
    rejected_conclusions: tuple[str, ...]


def parse_activity_response(raw_text: str, graph: SceneGraph, *, empty_confidence: float) -> ParsedActivityReasoning:
    """Parse Ollama JSON into evidence-backed ActivityHints."""
    payload = _extract_json(raw_text)
    activity = str(payload.get("activity", "")).strip()
    confidence_raw = payload.get("confidence", 0.5)
    confidence = max(
        0.0,
        min(1.0, float(confidence_raw) if isinstance(confidence_raw, (int, float, str)) else 0.5),
    )
    evidence_lines = _string_list(payload.get("evidence"))
    rejected = tuple(_string_list(payload.get("rejected_conclusions")))
    raw_indices = _int_list(payload.get("supporting_object_indices"))
    relation_types = tuple(_string_list(payload.get("supporting_relation_types")))
    caption = str(payload.get("caption", "")).strip()

    valid_indices = {node.index for node in graph.nodes}
    node_indices: tuple[int, ...] = tuple(
        index for index in raw_indices if index in valid_indices
    )
    if not node_indices and graph.nodes:
        node_indices = tuple(node.index for node in graph.nodes[:3])

    rationale = "; ".join(evidence_lines) if evidence_lines else "LLM evidence not provided."
    if not activity:
        activity = "static scene"
        confidence = empty_confidence

    activities = ActivityHints(
        activities=(
            ActivityEvidence(
                activity=activity,
                confidence=confidence,
                supporting_node_indices=node_indices,
                supporting_relation_types=relation_types,
                rationale=rationale,
            ),
        ),
        confidence=confidence,
    )
    return ParsedActivityReasoning(activities=activities, caption=caption, rejected_conclusions=rejected)


def _extract_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(cleaned)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Ollama response is not valid JSON")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result
