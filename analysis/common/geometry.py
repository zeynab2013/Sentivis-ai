"""Shared geometry helpers for scene analysis."""

import math

from core.contracts.detection import BoundingBox, DetectionResult


def normalized_center(box: BoundingBox, image_width: int, image_height: int) -> tuple[float, float]:
    """Return normalized center coordinates in [0, 1]."""
    return (
        box.center_x / max(1, image_width),
        box.center_y / max(1, image_height),
    )


def euclidean_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return pixel distance between bounding box centers."""
    dx = box_a.center_x - box_b.center_x
    dy = box_a.center_y - box_b.center_y
    return math.hypot(dx, dy)


def intersection_over_union(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return IoU between two axis-aligned boxes."""
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    union = box_a.area + box_b.area - intersection
    return intersection / max(union, 1.0)


def is_near(
    box_a: BoundingBox,
    box_b: BoundingBox,
    image_diagonal: float,
    near_ratio: float,
) -> bool:
    """Return True when box centers are within a fraction of image diagonal."""
    return euclidean_distance(box_a, box_b) <= image_diagonal * near_ratio


def is_inside(outer: BoundingBox, inner: BoundingBox, coverage: float = 0.8) -> bool:
    """Return True when inner box is mostly contained in outer box."""
    x_min = max(outer.x_min, inner.x_min)
    y_min = max(outer.y_min, inner.y_min)
    x_max = min(outer.x_max, inner.x_max)
    y_max = min(outer.y_max, inner.y_max)
    inter_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    return inter_area / max(inner.area, 1.0) >= coverage


def position_zone(
    box: BoundingBox,
    image_width: int,
    image_height: int,
    zone_split_low: float,
    zone_split_high: float,
) -> str:
    """Return vertical-horizontal zone label for a bounding box."""
    norm_x = box.center_x / max(1, image_width)
    norm_y = box.center_y / max(1, image_height)
    horizontal = "left" if norm_x < zone_split_low else "center" if norm_x < zone_split_high else "right"
    vertical = "top" if norm_y < zone_split_low else "middle" if norm_y < zone_split_high else "bottom"
    return f"{vertical}-{horizontal}"


def image_diagonal(result: DetectionResult) -> float:
    """Return image diagonal length in pixels."""
    return math.hypot(result.image_width, result.image_height)
