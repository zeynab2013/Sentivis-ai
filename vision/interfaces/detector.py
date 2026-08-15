"""Object detector interface."""

from typing import Protocol

from core.contracts.detection import DetectionResult
from core.contracts.image import PreprocessedImage


class IObjectDetector(Protocol):
    """Contract for object detection engines."""

    def detect(self, image: PreprocessedImage) -> DetectionResult:
        """Run detection and return structured results."""
        ...
