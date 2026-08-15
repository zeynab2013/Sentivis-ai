"""Color attributes must flow into natural captions when confident."""

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
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService


@pytest.fixture(autouse=True)
def _force_english_ui_language() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _StubVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="a scene", source="stub", confidence=0.4)


def _image() -> PreprocessedImage:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("color.jpg"),
        width=32,
        height=32,
        format_name="JPEG",
        size_bytes=100,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=32,
        inference_height=32,
        original_display_pixels=pixels,
    )


def _context() -> SceneContext:
    node = SceneNode(0, "obj-0", "person", 0.25, "middle-center")
    return SceneContext(
        graph=SceneGraph(nodes=(node,), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.5),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="park",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="casual",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="A person is present.",
    )


def test_clothing_color_promoted_into_caption() -> None:
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("person #1", "clothing_color", "red", 0.82, "attributes"),
            EvidenceFact("person #1", "action", "standing", 0.7, "pose_estimator"),
            EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.7, "environment"),
        ),
        ranked_subjects=("person #1",),
        environment_keys=("indoor_outdoor=outdoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person #1: clothing_color=red",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    paragraph = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), understanding, context=_context()
    )
    assert "red" in paragraph.lower()


def test_object_dominant_color_in_caption() -> None:
    car = SceneNode(1, "obj-1", "car", 0.3, "middle-center")
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("car #1", "is", "car", 0.9, "yolo"),
            EvidenceFact("car #1", "dominant_color", "blue", 0.8, "attributes"),
            EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.7, "environment"),
        ),
        ranked_subjects=("car #1",),
        environment_keys=("indoor_outdoor=outdoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="car #1: dominant_color=blue",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    context = SceneContext(
        graph=SceneGraph(nodes=(car,), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.5),
        environment=EnvironmentInfo(
            scene_type="street",
            setting="street",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="public",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("car",),
        spatial_summary="A car is present.",
    )
    paragraph = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), understanding, context=context
    )
    assert "blue" in paragraph.lower()
