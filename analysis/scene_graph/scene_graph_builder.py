"""Build scene graph from detections and relations."""

from analysis.common.geometry import position_zone
from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import Relation, SceneGraph, SceneNode
from core.contracts.detection import DetectionResult
from core.logging import get_logger

logger = get_logger(__name__)


class SceneGraphBuilder:
    """Constructs a scene graph from detections and relations."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config

    def build(
        self,
        detections: DetectionResult,
        relations: tuple[Relation, ...],
    ) -> SceneGraph:
        image_area = max(1.0, detections.image_width * detections.image_height)
        cfg = self._config.attributes
        nodes: list[SceneNode] = []
        for index, detection in enumerate(detections.detections):
            area_ratio = detection.bounding_box.area / image_area
            zone = position_zone(
                detection.bounding_box,
                detections.image_width,
                detections.image_height,
                cfg.zone_split_low,
                cfg.zone_split_high,
            )
            nodes.append(
                SceneNode(
                    index=index,
                    object_id=detection.object_id,
                    label=detection.label,
                    bounding_box_area_ratio=area_ratio,
                    position_zone=zone,
                )
            )
        graph = SceneGraph(nodes=tuple(nodes), relations=relations)
        logger.debug("Built scene graph with %d nodes", len(nodes))
        return graph
