"""Unit tests for object crop and clothing analysis."""

import numpy as np

from analysis.clothing.clothing_analyzer import ClothingAnalyzer
from core.contracts.detection import BoundingBox, Detection
from vision.crop_analysis.object_crop_analyzer import ObjectCropAnalyzer


def test_clothing_analyzer_returns_garment_colors() -> None:
    pixels = np.zeros((120, 60, 3), dtype=np.uint8)
    pixels[:40, :] = (20, 20, 20)
    pixels[40:80, :] = (30, 60, 180)
    pixels[80:, :] = (20, 20, 20)
    box = BoundingBox(0, 0, 60, 120)
    result = ClothingAnalyzer().analyze(pixels, box, detection_confidence=0.9)
    assert result is not None
    assert result.shirt_color in {"blue", "cyan", "purple", "navy blue", "sky blue"}
    assert result.clothing_type
    assert result.clothing_style
    assert result.dominant_colors
    assert result.confidence >= 0.55


def test_object_crop_analyzer_person_description() -> None:
    pixels = np.full((160, 80, 3), 90, dtype=np.uint8)
    pixels[40:100, :] = (40, 90, 200)
    detection = Detection(
        object_id="obj-0",
        label="person",
        confidence=0.95,
        bounding_box=BoundingBox(0, 0, 80, 160),
        class_id=0,
        detected_at=0.0,
    )
    crop = ObjectCropAnalyzer().analyze(pixels, detection)
    assert crop.clothing is not None
    assert "person" in crop.description.lower()
    assert crop.dominant_color != "unknown"
