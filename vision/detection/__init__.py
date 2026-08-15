"""Object detection package."""

from vision.detection.detector_service import ManagedObjectDetector
from vision.detection.yolo_engine import YoloEngine

__all__ = ["ManagedObjectDetector", "YoloEngine"]
