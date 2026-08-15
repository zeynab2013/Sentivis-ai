"""No-op object tracker for v1."""

from core.contracts.detection import DetectionResult


class NoOpTracker:
    """Returns detections unchanged until tracking is implemented."""

    def track(self, detections: DetectionResult) -> DetectionResult:
        """Pass through detections without modification."""
        return detections
