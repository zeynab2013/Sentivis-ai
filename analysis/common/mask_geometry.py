"""Mask geometry helpers for segmentation-aware reasoning."""

from __future__ import annotations

from core.contracts.detection import BoundingBox, Detection, SegmentationMask


def polygon_area(polygon: tuple[tuple[float, float], ...]) -> float:
    """Shoelace formula for polygon area."""
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def mask_centroid(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    """Compute centroid of polygon."""
    if not polygon:
        return (0.0, 0.0)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def mask_overlap_ratio(
    mask_a: SegmentationMask,
    mask_b: SegmentationMask,
    image_area: float,
) -> float:
    """Approximate overlap using bounding boxes of polygons when full raster unavailable."""
    if not mask_a.polygon or not mask_b.polygon:
        return 0.0
    box_a = _polygon_bounds(mask_a.polygon)
    box_b = _polygon_bounds(mask_b.polygon)
    inter = _box_intersection(box_a, box_b)
    if inter <= 0 or image_area <= 0:
        return 0.0
    smaller = min(polygon_area(mask_a.polygon), polygon_area(mask_b.polygon))
    if smaller <= 0:
        return 0.0
    return min(1.0, inter / smaller)


def mask_contains(
    outer: SegmentationMask,
    inner: SegmentationMask,
    *,
    min_ratio: float = 0.55,
) -> bool:
    """Return True when inner mask centroid lies inside outer polygon bounds with area check."""
    if not outer.polygon or not inner.polygon:
        return False
    inner_area = polygon_area(inner.polygon)
    outer_area = polygon_area(outer.polygon)
    if inner_area <= 0 or outer_area <= inner_area:
        return False
    centroid = mask_centroid(inner.polygon)
    if not _point_in_polygon(centroid, outer.polygon):
        return False
    overlap = mask_overlap_ratio(outer, inner, outer_area + inner_area)
    return overlap >= min_ratio


def bbox_to_polygon(box: BoundingBox) -> tuple[tuple[float, float], ...]:
    """Convert bounding box to rectangular polygon."""
    return (
        (box.x_min, box.y_min),
        (box.x_max, box.y_min),
        (box.x_max, box.y_max),
        (box.x_min, box.y_max),
    )


def enrich_mask(
    mask: SegmentationMask,
    *,
    image_width: int,
    image_height: int,
) -> SegmentationMask:
    """Populate centroid, contour, visible percentage from polygon."""
    image_area = float(max(1, image_width * image_height))
    area = polygon_area(mask.polygon)
    centroid = mask_centroid(mask.polygon)
    visible = min(1.0, area / max(image_area * max(mask.area_ratio, 0.001), 1.0))
    return SegmentationMask(
        polygon=mask.polygon,
        area_ratio=mask.area_ratio if mask.area_ratio > 0 else area / image_area,
        centroid=centroid,
        contour=mask.contour or mask.polygon,
        visible_percentage=visible,
        occlusion_estimate=max(0.0, 1.0 - visible),
    )


def _polygon_bounds(polygon: tuple[tuple[float, float], ...]) -> BoundingBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def _box_intersection(a: BoundingBox, b: BoundingBox) -> float:
    x_min = max(a.x_min, b.x_min)
    y_min = max(a.y_min, b.y_min)
    x_max = min(a.x_max, b.x_max)
    y_max = min(a.y_max, b.y_max)
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-6) + x1):
            inside = not inside
    return inside


def detection_with_fallback_mask(detection: Detection, image_width: int, image_height: int) -> Detection:
    """Ensure detection has segmentation mask from YOLO or bbox fallback."""
    if detection.segmentation is not None and detection.segmentation.polygon:
        mask = enrich_mask(detection.segmentation, image_width=image_width, image_height=image_height)
        return Detection(
            object_id=detection.object_id,
            label=detection.label,
            confidence=detection.confidence,
            bounding_box=detection.bounding_box,
            class_id=detection.class_id,
            detected_at=detection.detected_at,
            segmentation=mask,
        )
    polygon = bbox_to_polygon(detection.bounding_box)
    image_area = float(max(1, image_width * image_height))
    area_ratio = detection.bounding_box.area / image_area
    mask = enrich_mask(
        SegmentationMask(polygon=polygon, area_ratio=area_ratio),
        image_width=image_width,
        image_height=image_height,
    )
    return Detection(
        object_id=detection.object_id,
        label=detection.label,
        confidence=detection.confidence,
        bounding_box=detection.bounding_box,
        class_id=detection.class_id,
        detected_at=detection.detected_at,
        segmentation=mask,
    )
