"""Production generalization: no scene/activity hallucination from weak co-presence."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    Relation,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.assistant.suggested_questions import _is_caption_duplicate
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService


@pytest.fixture(autouse=True)
def _english() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _QuietVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


def _image(name: str = "scene.jpg") -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path(name),
        width=48,
        height=48,
        format_name="JPEG",
        size_bytes=100,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=48,
        inference_height=48,
    )


def test_dining_table_does_not_become_restaurant() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("dining table #1", "is", "dining table", 0.9, "yolo"),
            EvidenceFact("cup #1", "is", "cup", 0.8, "yolo"),
            EvidenceFact("bowl #1", "is", "bowl", 0.8, "yolo"),
            EvidenceFact("person #1", "is", "person", 0.85, "yolo"),
            EvidenceFact("scene", "indoor_outdoor", "indoor", 0.8, "environment"),
            EvidenceFact("scene", "setting", "room", 0.7, "environment"),
        ),
        ranked_subjects=("person #1", "dining table #1", "cup #1", "bowl #1"),
        environment_keys=("indoor_outdoor=indoor", "setting=room"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person dining table cup bowl indoor",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    story = service._story_facts(understanding)
    label = service._concrete_scene_label(story)
    assert label != "restaurant"
    assert label in {"dining area", "room", "indoor space", ""}
    caption = service.generate(_image(), understanding)
    assert "restaurant" not in caption.lower()


def test_person_near_horse_does_not_invent_leading() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("horse #1", "is", "horse", 0.9, "yolo"),
            EvidenceFact("person #1", "near", "horse", 0.7, "relationships"),
            EvidenceFact("scene", "setting", "field", 0.8, "environment"),
        ),
        ranked_subjects=("person #1", "horse #1"),
        environment_keys=("setting=field",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person near horse field",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    people = ("person #1",)
    inferred = service._infer_rich_activity(understanding, people, "", "near horse")
    assert "leading" not in (inferred or "").lower()
    assert "working with a horse" not in (inferred or "").lower()
    caption = service.generate(_image(), understanding)
    lower = caption.lower()
    assert "leading" not in lower
    assert "horse" in lower


def test_fridge_alone_does_not_force_cooking() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("refrigerator #1", "is", "refrigerator", 0.9, "yolo"),
            EvidenceFact("scene", "setting", "kitchen", 0.8, "environment"),
        ),
        ranked_subjects=("person #1", "refrigerator #1"),
        environment_keys=("setting=kitchen",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person refrigerator kitchen",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    inferred = service._infer_rich_activity(understanding, ("person #1",), "", "")
    assert (inferred or "").lower() not in {"cooking", "preparing food"}


def test_color_question_blocked_when_caption_already_states_color() -> None:
    caption = "Two brown chairs surround a wooden dining table."
    assert _is_caption_duplicate(
        "What color are the chairs?",
        caption,
        ("brown", "chair"),
    )
    assert _is_caption_duplicate(
        "What color is the chair?",
        caption,
        ("brown",),
    )


def test_kitchen_with_prep_keeps_activity() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("person #1", "action", "kitchen preparation", 0.8, "pose_estimator"),
            EvidenceFact("refrigerator #1", "is", "refrigerator", 0.85, "yolo"),
            EvidenceFact("dining table #1", "is", "dining table", 0.8, "yolo"),
            EvidenceFact("scene", "setting", "kitchen", 0.85, "environment"),
        ),
        ranked_subjects=("person #1", "refrigerator #1", "dining table #1"),
        environment_keys=("setting=kitchen",),
        activity_keys=("kitchen preparation",),
        ocr_text=(),
        evidence_brief="person kitchen preparation",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    caption = service.generate(_image(), understanding)
    lower = caption.lower()
    assert "restaurant" not in lower
    assert any(tok in lower for tok in ("kitchen", "prepar", "food", "person"))
