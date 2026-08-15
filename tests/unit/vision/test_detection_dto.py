"""Unit tests for detection DTO construction."""

import time

from core.contracts.detection import BoundingBox, Detection, DetectionResult, SegmentationMask


def test_detection_carries_object_id_and_timestamp() -> None:
    now = time.time()
    detection = Detection(
        object_id="obj-abc",
        label="person",
        confidence=0.91,
        bounding_box=BoundingBox(1, 2, 3, 4),
        class_id=0,
        detected_at=now,
        segmentation=SegmentationMask(polygon=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)), area_ratio=0.1),
    )
    result = DetectionResult(
        detections=(detection,),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    assert result.detections[0].object_id == "obj-abc"
    assert result.detections[0].segmentation is not None
    assert result.inference_timestamp == now
