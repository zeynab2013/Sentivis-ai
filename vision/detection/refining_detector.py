"""Object detector that chains YOLO detection with SAM2 mask refinement."""

from __future__ import annotations

from core.config.app_config import AppConfig
from core.config.model_config import ModelConfig
from core.contracts.detection import DetectionResult
from core.contracts.image import PreprocessedImage
from core.logging import get_logger
from vision.detection.detector_service import ManagedObjectDetector
from vision.segmentation.sam2_refiner import Sam2SegmentationRefiner

logger = get_logger(__name__)


class RefiningObjectDetector:
    """YOLO detection followed by optional SAM2 boundary refinement."""

    def __init__(
        self,
        base_detector: ManagedObjectDetector,
        app_config: AppConfig,
        model_config: ModelConfig,
    ) -> None:
        self._base = base_detector
        self._refiner = Sam2SegmentationRefiner(app_config.paths.models_dir)
        self._model_config = model_config
        self._enable_sam2 = True

    def set_sam2_enabled(self, enabled: bool) -> None:
        self._enable_sam2 = enabled

    def detect(self, image: PreprocessedImage) -> DetectionResult:
        result = self._base.detect(image)
        if not self._enable_sam2:
            return result
        refined = self._refiner.refine(result, image)
        logger.info(
            "Segmentation refinement complete (sam2=%s) for %d objects",
            self._refiner.available,
            len(refined.detections),
        )
        return refined
