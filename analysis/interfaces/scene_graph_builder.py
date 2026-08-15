"""Scene graph builder interface."""

from typing import Protocol

from core.contracts.analysis import Relation, SceneGraph
from core.contracts.detection import DetectionResult


class ISceneGraphBuilder(Protocol):
    """Build scene graph from detections and relations."""

    def build(
        self,
        detections: DetectionResult,
        relations: tuple[Relation, ...],
    ) -> SceneGraph:
        """Return scene graph."""
        ...
