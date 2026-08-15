"""Regression: broken Florence VLM must not survive via fragment concatenation.

Reproduces the Streamlit failure caption shape:
VLM inventory English + "Wooden chairs around the table." + "X is also nearby."
+ "A refrigerator, 2 brown chairs, and a couch are visible in a kitchen."
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    Attribute,
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
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService


@pytest.fixture(autouse=True)
def _english() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


_BROKEN_FLORENCE = (
    "A man and a woman standing on the floor and in front of them A table and on table "
    "we have a flower vase, plate with fruits, cup, tissue box and in the background we can find "
    "a refrigerator, cupboards, oven, stove, bowls, plants, television, wall, clock, curtains, "
    "window and some objects."
)

_FAILURE_CONCAT = (
    f"{_BROKEN_FLORENCE} Wooden chairs around the table. "
    "A dining table is also nearby. A couch is also nearby. A sink is also nearby. "
    "A refrigerator, 2 brown chairs, and a couch are visible in a kitchen."
)


class _BrokenFlorenceVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text=_BROKEN_FLORENCE, source="florence_base", confidence=0.7)


def _image() -> PreprocessedImage:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("kitchen_dining.png"),
        width=64,
        height=64,
        format_name="PNG",
        size_bytes=100,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=64,
        inference_height=64,
        original_display_pixels=pixels,
    )


def _kitchen_understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
        EvidenceFact("person #2", "is", "person", 0.85, "yolo"),
        EvidenceFact("dining table #1", "is", "dining table", 0.9, "yolo"),
        EvidenceFact("chair #1", "is", "chair", 0.9, "yolo"),
        EvidenceFact("chair #1", "dominant_color", "brown", 0.8, "attributes"),
        EvidenceFact("chair #2", "is", "chair", 0.88, "yolo"),
        EvidenceFact("chair #2", "dominant_color", "brown", 0.8, "attributes"),
        EvidenceFact("refrigerator #1", "is", "refrigerator", 0.91, "yolo"),
        EvidenceFact("couch #1", "is", "couch", 0.88, "yolo"),
        EvidenceFact("sink #1", "is", "sink", 0.7, "yolo"),
        EvidenceFact("vase #1", "is", "vase", 0.89, "yolo"),
        EvidenceFact("cup #1", "is", "cup", 0.89, "yolo"),
        EvidenceFact("scene", "setting", "kitchen", 0.85, "environment"),
        EvidenceFact(
            "chair #1",
            "spatial",
            "wooden chairs around the table",
            0.75,
            "relationships",
        ),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=(
            "person #1",
            "person #2",
            "dining table #1",
            "chair #1",
            "chair #2",
            "refrigerator #1",
            "couch #1",
            "sink #1",
            "vase #1",
            "cup #1",
        ),
        environment_keys=("setting=kitchen",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="two people dining table chairs refrigerator kitchen",
        overall_confidence=0.85,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _kitchen_context() -> SceneContext:
    nodes = (
        SceneNode(0, "obj-0", "person", 0.09, "middle-center"),
        SceneNode(1, "obj-1", "person", 0.04, "middle-center"),
        SceneNode(2, "obj-2", "dining table", 0.30, "middle-center"),
        SceneNode(3, "obj-3", "chair", 0.03, "middle-left"),
        SceneNode(4, "obj-4", "chair", 0.03, "middle-right"),
        SceneNode(5, "obj-5", "refrigerator", 0.06, "middle-right"),
        SceneNode(6, "obj-6", "couch", 0.04, "middle-left"),
        SceneNode(7, "obj-7", "sink", 0.01, "background"),
        SceneNode(8, "obj-8", "vase", 0.01, "middle-center"),
        SceneNode(9, "obj-9", "cup", 0.01, "middle-center"),
    )
    return SceneContext(
        graph=SceneGraph(
            nodes=nodes,
            relations=(
                Relation(0, 2, "near", 0.7),
                Relation(1, 2, "near", 0.7),
                Relation(3, 2, "around", 0.7),
            ),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(3, "dominant_color", "brown"),
                Attribute(4, "dominant_color", "brown"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    "standing",
                    0.5,
                    (0, 1, 2),
                    ("near",),
                    "people near dining table",
                ),
            ),
            confidence=0.5,
        ),
        environment=EnvironmentInfo(
            scene_type="kitchen",
            setting="kitchen",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="home",
            crowd_level="sparse",
            scene_complexity="complex",
            evidence=("indoor kitchen",),
        ),
        object_count=10,
        dominant_objects=("person", "dining table", "chair", "refrigerator", "couch"),
        spatial_summary="Two people near a dining table in a kitchen.",
    )


def test_broken_florence_english_rejected_as_spine() -> None:
    service = NaturalCaptionService(_BrokenFlorenceVision())  # type: ignore[arg-type]
    assert service._is_broken_natural_english(_BROKEN_FLORENCE)
    assert not service._is_broken_natural_english(
        "Two people are in a kitchen around a dining table with chairs."
    )


def test_failure_concat_shape_rejected_by_fragment_spam() -> None:
    service = NaturalCaptionService(_BrokenFlorenceVision())  # type: ignore[arg-type]
    assert service._caption_has_fragment_spam(_FAILURE_CONCAT)
    assert service._is_caption_fragment("Wooden chairs around the table.")
    assert service._is_caption_fragment("A dining table is also nearby.")


def test_gate_strips_failure_concat_shape() -> None:
    service = NaturalCaptionService(_BrokenFlorenceVision())  # type: ignore[arg-type]
    understanding = _kitchen_understanding()
    story = service._story_facts(understanding)
    cleaned = service._final_naturalness_gate(_FAILURE_CONCAT, understanding, story)
    lower = cleaned.lower()
    assert "we have" not in lower
    assert "we can find" not in lower
    assert "standing on the floor" not in lower
    assert "is also nearby" not in lower
    assert "some objects" not in lower
    assert "wooden chairs around the table." not in lower


def test_generate_does_not_emit_failure_concat_shape() -> None:
    caption = NaturalCaptionService(_BrokenFlorenceVision()).generate(  # type: ignore[arg-type]
        _image(), _kitchen_understanding(), context=_kitchen_context()
    )
    lower = caption.lower()
    assert "we have" not in lower
    assert "we can find" not in lower
    assert "standing on the floor" not in lower
    assert "is also nearby" not in lower
    assert "some objects" not in lower
    assert "are visible in a kitchen" not in lower
    assert lower.count("nearby") == 0
    assert "close by" not in lower
    assert "arranged close" not in lower
    assert lower.count("dining table") <= 1
    assert lower.count("refrigerator") <= 1
    assert lower.count("couch") <= 1
    # Must remain a coherent multi-sentence paragraph with people + table context.
    assert any(tok in lower for tok in ("person", "people", "man", "woman"))
    assert "table" in lower or "kitchen" in lower
    assert any(tok in lower for tok in ("vase", "cup", "chair"))
    assert caption.endswith(".")
    assert len(caption.split()) >= 20
    # Spatial support should not be a flat object dump.
    assert "are visible in a kitchen" not in lower
    assert not re.search(r"\b\d+\s+refrigerators?\b", lower)
