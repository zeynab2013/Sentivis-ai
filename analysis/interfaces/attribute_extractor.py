"""Attribute extractor interface."""

from typing import Protocol

from core.contracts.analysis import AttributeSet
from core.contracts.detection import DetectionResult
from core.contracts.image import PreprocessedImage


class IAttributeExtractor(Protocol):
    """Extract object attributes from detections and image pixels."""

    def extract(self, detections: DetectionResult, image: PreprocessedImage) -> AttributeSet:
        """Return attributes for each detected object."""
        ...
