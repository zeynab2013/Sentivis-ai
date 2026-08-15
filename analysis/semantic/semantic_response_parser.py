"""Parse Ollama semantic synthesis JSON response."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedSemanticReasoning:
    """Validated semantic synthesis output."""

    caption: str
    scene_explanation: str
    rejected_conclusions: tuple[str, ...]
    contradictions_resolved: tuple[str, ...]


def parse_semantic_response(raw_text: str) -> ParsedSemanticReasoning:
    payload = _extract_json(raw_text)
    caption = str(payload.get("caption", "")).strip()
    explanation = str(payload.get("scene_explanation", "")).strip()
    rejected = tuple(_string_list(payload.get("rejected_conclusions")))
    resolved = tuple(_string_list(payload.get("contradictions_resolved")))
    if not caption and explanation:
        caption = explanation
    return ParsedSemanticReasoning(
        caption=caption,
        scene_explanation=explanation,
        rejected_conclusions=rejected,
        contradictions_resolved=resolved,
    )


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
    raise ValueError("Ollama semantic response is not valid JSON")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
