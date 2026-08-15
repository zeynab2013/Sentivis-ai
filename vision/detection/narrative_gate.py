"""Narrative confidence gating for detections.

A detection may exist for visualization without being trusted by the language layer.
"""

from __future__ import annotations

from core.contracts.detection import Detection

# Objects below these floors may still appear in the detector overlay but must
# not be promoted into caption/QA facts without a second evidence source.
_NARRATIVE_MIN_CONFIDENCE: dict[str, float] = {
    "person": 0.32,
    "horse": 0.34,
    "dog": 0.34,
    "cat": 0.34,
    "car": 0.36,
    "bicycle": 0.36,
    "chair": 0.40,
    "couch": 0.40,
    "dining table": 0.40,
    "laptop": 0.45,
    "cell phone": 0.72,
    "bottle": 0.58,
    "cup": 0.58,
    "book": 0.58,
    "backpack": 0.78,
    "handbag": 0.75,
    "tie": 0.75,
}

_DEFAULT_NARRATIVE_MIN = 0.45


def narrative_min_confidence(label: str) -> float:
    return _NARRATIVE_MIN_CONFIDENCE.get((label or "").lower().strip(), _DEFAULT_NARRATIVE_MIN)


def is_narrative_reliable(detection: Detection) -> bool:
    """True when a detection is strong enough for caption/QA fact use."""
    return float(detection.confidence) >= narrative_min_confidence(detection.label)


def filter_narrative_detections(detections: tuple[Detection, ...] | list[Detection]) -> tuple[Detection, ...]:
    """Keep only narrative-reliable detections (visualization may use the full set)."""
    return tuple(det for det in detections if is_narrative_reliable(det))
