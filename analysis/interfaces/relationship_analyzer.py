"""Relationship analyzer interface."""

from typing import Protocol

from core.contracts.analysis import Relation
from core.contracts.detection import DetectionResult


class IRelationshipAnalyzer(Protocol):
    """Analyze spatial relationships between objects."""

    def analyze(self, detections: DetectionResult) -> tuple[Relation, ...]:
        """Return pairwise relations."""
        ...
