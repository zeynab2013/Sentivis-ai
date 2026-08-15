"""Accessory confidence filtering for YOLO post-processing."""

from __future__ import annotations

from core.config.model_config import YoloModelConfig
from core.contracts.detection import BoundingBox, Detection
from vision.detection.yolo_engine import YoloEngine


def _engine() -> YoloEngine:
    return YoloEngine(
        YoloModelConfig(
            variant="yolo11n",
            weights_path=None,
            confidence_threshold=0.35,
            iou_threshold=0.45,
            preferred_device="cpu",
        )
    )


def test_low_confidence_backpack_is_removed() -> None:
    engine = _engine()
    raw = [
        Detection(
            object_id="a",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(0, 0, 100, 200),
            class_id=0,
            detected_at=0.0,
        ),
        Detection(
            object_id="b",
            label="backpack",
            confidence=0.45,
            bounding_box=BoundingBox(10, 40, 40, 90),
            class_id=24,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=1000 * 1000)
    labels = {item.label for item in kept}
    assert "person" in labels
    assert "backpack" not in labels
    assert any(item[0] == "backpack" for item in removed)


def test_high_confidence_backpack_kept() -> None:
    engine = _engine()
    raw = [
        Detection(
            object_id="b",
            label="backpack",
            confidence=0.86,
            bounding_box=BoundingBox(10, 40, 120, 220),
            class_id=24,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=1000 * 1000)
    assert len(kept) == 1
    assert kept[0].label == "backpack"
    assert removed == []


def test_confusable_chair_bench_keeps_one() -> None:
    engine = _engine()
    raw = [
        Detection(
            object_id="c",
            label="chair",
            confidence=0.55,
            bounding_box=BoundingBox(100, 100, 180, 220),
            class_id=56,
            detected_at=0.0,
        ),
        Detection(
            object_id="b",
            label="bench",
            confidence=0.52,
            bounding_box=BoundingBox(105, 110, 200, 210),
            class_id=13,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=1000 * 1000)
    labels = {item.label for item in kept}
    assert len(labels & {"chair", "bench"}) == 1
    assert any("confusable" in item[2] for item in removed)


def test_duplicate_same_label_suppressed() -> None:
    engine = _engine()
    raw = [
        Detection(
            object_id="a",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(10, 10, 120, 240),
            class_id=0,
            detected_at=0.0,
        ),
        Detection(
            object_id="b",
            label="person",
            confidence=0.85,
            bounding_box=BoundingBox(20, 20, 125, 245),
            class_id=0,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=1000 * 1000)
    assert len([item for item in kept if item.label == "person"]) == 1
    assert any(item[2] == "duplicate_semantic" for item in removed)


def test_primary_subject_kept_below_global_threshold() -> None:
    engine = _engine()
    raw = [
        Detection(
            object_id="p",
            label="person",
            confidence=0.30,
            bounding_box=BoundingBox(0, 0, 120, 240),
            class_id=0,
            detected_at=0.0,
        ),
        Detection(
            object_id="c",
            label="chair",
            confidence=0.36,
            bounding_box=BoundingBox(200, 200, 280, 320),
            class_id=56,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=1000 * 1000)
    labels = {item.label for item in kept}
    assert "person" in labels
    assert "chair" not in labels
    assert any(item[0] == "chair" for item in removed)
