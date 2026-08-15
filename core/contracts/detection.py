"""Object detection DTOs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Return box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return box height."""
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        """Return horizontal center."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        """Return vertical center."""
        return (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        """Return box area."""
        return max(0.0, self.width * self.height)


@dataclass(frozen=True)
class SegmentationMask:
    """Polygon segmentation for a detected object in source-image coordinates."""

    polygon: tuple[tuple[float, float], ...]
    area_ratio: float
    centroid: tuple[float, float] | None = None
    contour: tuple[tuple[float, float], ...] | None = None
    visible_percentage: float | None = None
    occlusion_estimate: float | None = None

@dataclass(frozen=True)
class Detection:
    """Single detected object."""

    object_id: str
    label: str
    confidence: float
    bounding_box: BoundingBox
    class_id: int
    detected_at: float
    segmentation: SegmentationMask | None = None


@dataclass(frozen=True)
class DetectionResult:
    """Complete detection output for an image."""

    detections: tuple[Detection, ...]
    image_width: int
    image_height: int
    inference_timestamp: float
