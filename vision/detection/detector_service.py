"""Model-backed object detection service."""

from typing import Protocol

from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from core.contracts.detection import DetectionResult
from core.contracts.image import PreprocessedImage
from core.logging import get_logger
from services.interfaces.model_manager import IModelManager

logger = get_logger(__name__)


class _InferenceEngine(Protocol):
    """Internal inference surface for acquired detection engines."""

    def infer(self, image: PreprocessedImage) -> DetectionResult:
        ...


class ManagedObjectDetector:
    """Detects objects using ModelManager lifecycle."""

    def __init__(self, model_manager: IModelManager, model_config: ModelConfig) -> None:
        self._model_manager = model_manager
        self._preferred_device = model_config.yolo.preferred_device

    def detect(self, image: PreprocessedImage) -> DetectionResult:
        """Acquire detection engine, infer, and release."""
        engine = self._model_manager.acquire(ModelKind.YOLO, self._preferred_device)
        try:
            inference_engine: _InferenceEngine = engine  # type: ignore[assignment]
            result = inference_engine.infer(image)
            logger.info("Detection completed with %d objects", len(result.detections))
            return result
        finally:
            self._model_manager.release_active()
