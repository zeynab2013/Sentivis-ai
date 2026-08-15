"""Object tracker interface (future-ready)."""

from typing import Protocol

from core.contracts.detection import DetectionResult


class IObjectTracker(Protocol):
    """Contract for multi-frame object tracking."""

    def track(self, detections: DetectionResult) -> DetectionResult:
        """Return detections with temporal IDs applied."""
        ...
