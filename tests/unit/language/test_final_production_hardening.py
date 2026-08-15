"""FINAL production hardening regressions: caption grammar + STOP/OCR dedupe."""

from __future__ import annotations

import numpy as np

from analysis.ocr.text_extractor import OcrExtractor
from core.contracts.detection import BoundingBox, Detection
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.refinement.caption_sanity import sanitize_caption
from language.validation.caption_factuality import clamp_caption_object_counts
from vision.detection.yolo_engine import YoloEngine
from core.config.model_config import YoloModelConfig


def test_malformed_outdoor_activity_caption_repaired() -> None:
    raw = (
        "Two people are an outdoor activity involving a tan horse. "
        "The horse's dominant color is tan, and it is moving with purpose. "
        "A fire is burning nearby. A person is leading a horse. A person is holding a rope."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "are an outdoor activity" not in lower
    assert "are an activity" not in lower
    assert "dominant color is" not in lower
    assert "observed activity:" not in lower
    assert "horse" in lower
    assert "fire" in lower or "burning" in lower
    assert "leading" in lower or "holding" in lower or "rope" in lower


def test_metadata_leakage_stripped() -> None:
    raw = (
        "Person, and bicycle. The location is outdoor. "
        "Observed activity: riding a bicycle."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "observed activity:" not in lower
    assert "location is outdoor" not in lower
    assert "person, and bicycle" not in lower


def test_nested_stop_sign_boxes_deduped_to_one() -> None:
    engine = YoloEngine(
        YoloModelConfig(
            variant="yolo11n",
            weights_path=None,
            confidence_threshold=0.35,
            iou_threshold=0.45,
            preferred_device="cpu",
        )
    )
    raw = [
        Detection(
            object_id="a",
            label="stop sign",
            confidence=0.94,
            bounding_box=BoundingBox(40, 40, 200, 220),
            class_id=11,
            detected_at=0.0,
        ),
        Detection(
            object_id="b",
            label="stop sign",
            confidence=0.88,
            bounding_box=BoundingBox(55, 55, 180, 200),
            class_id=11,
            detected_at=0.0,
        ),
        Detection(
            object_id="c",
            label="stop sign",
            confidence=0.80,
            bounding_box=BoundingBox(70, 70, 160, 180),
            class_id=11,
            detected_at=0.0,
        ),
        Detection(
            object_id="d",
            label="stop sign",
            confidence=0.76,
            bounding_box=BoundingBox(50, 50, 190, 210),
            class_id=11,
            detected_at=0.0,
        ),
    ]
    kept, removed = engine._filter_detections(raw, image_area=640 * 480)
    assert len([d for d in kept if d.label == "stop sign"]) == 1
    assert any(item[2] == "duplicate_semantic" for item in removed)


def test_caption_clamps_four_stop_signs_to_verified_one() -> None:
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("stop_sign_1", "label", "stop sign", 0.9, "yolo"),
        ),
        ranked_subjects=("stop_sign_1",),
        environment_keys=(),
        activity_keys=(),
        ocr_text=("STOP",),
        evidence_brief="stop_sign_1",
        overall_confidence=0.9,
        discarded_count=0,
        contradictions_resolved=0,
    )
    text = "4 stop signs are nearby. The readable text is STOP."
    cleaned = clamp_caption_object_counts(text, understanding)
    lower = cleaned.lower()
    assert "4 stop signs" not in lower
    assert "four stop signs" not in lower
    assert "stop sign" in lower
    assert "stop" in lower


def test_ocr_dedupe_repeated_tokens() -> None:
    deduped = OcrExtractor._dedupe_ocr_texts(("STOP", "STOP", "stop", "S", "T"))
    upper = [t.upper() for t in deduped]
    assert upper.count("STOP") == 1


def test_ocr_orientation_retry_flag_for_fragments() -> None:
    assert OcrExtractor._needs_orientation_retry((), 0.0) is True
    assert OcrExtractor._needs_orientation_retry(("S", "T", "O", "P"), 0.6) is True
    assert OcrExtractor._needs_orientation_retry(("STOP",), 0.9) is False
