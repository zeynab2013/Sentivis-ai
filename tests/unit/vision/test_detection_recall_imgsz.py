"""Regression: YOLO must use prepared inference resolution (not silent 640 downscale)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from core.config.model_config import YoloModelConfig
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage, ValidatedImage
from vision.detection.yolo_engine import YoloEngine
from analysis.attributes.attribute_extractor import AttributeExtractor
from core.config.loader import load_analysis_config


def _preprocessed(*, infer: int = 1280, source: int = 1600) -> PreprocessedImage:
    display = np.zeros((source, source, 3), dtype=np.uint8)
    inference = np.zeros((infer, infer, 3), dtype=np.uint8)
    validated = ValidatedImage(
        path=__file__,
        width=source,
        height=source,
        format_name="png",
        size_bytes=source * source * 3,
        pixels=display,
    )
    return PreprocessedImage(
        source=validated,
        display_pixels=display,
        inference_pixels=inference,
        inference_width=infer,
        inference_height=infer,
        original_display_pixels=display,
    )


def test_yolo_predict_uses_inference_imgsz_not_default_640() -> None:
    engine = YoloEngine(
        YoloModelConfig(
            variant="yolo11n",
            weights_path=None,
            confidence_threshold=0.35,
            iou_threshold=0.45,
            preferred_device="cpu",
        )
    )
    mock_model = MagicMock()
    empty = SimpleNamespace(boxes=None, masks=None, names={})
    mock_model.predict.return_value = [empty]
    engine._model = mock_model
    engine._loaded = True
    engine._device = "cpu"

    engine.infer(_preprocessed(infer=1280, source=1600))

    kwargs = mock_model.predict.call_args.kwargs
    assert "imgsz" in kwargs
    assert kwargs["imgsz"] == 1280
    assert kwargs["imgsz"] != 640


def test_attribute_crops_remap_when_original_resolution_differs() -> None:
    """SR/enhanced detection space must be scaled back onto original pixels."""
    extractor = AttributeExtractor(load_analysis_config())
    # Detection space is 2× original.
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="p1",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(100, 200, 300, 600),
                class_id=0,
                detected_at=0.0,
            ),
        ),
        image_width=800,
        image_height=800,
        inference_timestamp=0.0,
    )
    original = np.zeros((400, 400, 3), dtype=np.uint8)
    aligned = extractor._align_detections_to_pixels(detections, original)
    assert len(aligned) == 1
    box = aligned[0].bounding_box
    assert abs(box.x_min - 50.0) < 1e-6
    assert abs(box.y_min - 100.0) < 1e-6
    assert abs(box.x_max - 150.0) < 1e-6
    assert abs(box.y_max - 300.0) < 1e-6


def test_attribute_alignment_noop_when_resolutions_match() -> None:
    extractor = AttributeExtractor(load_analysis_config())
    det = Detection(
        object_id="p1",
        label="person",
        confidence=0.9,
        bounding_box=BoundingBox(10, 20, 30, 40),
        class_id=0,
        detected_at=0.0,
    )
    detections = DetectionResult(
        detections=(det,),
        image_width=100,
        image_height=100,
        inference_timestamp=0.0,
    )
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    aligned = extractor._align_detections_to_pixels(detections, pixels)
    assert aligned[0].bounding_box == det.bounding_box
