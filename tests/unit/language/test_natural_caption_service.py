"""Tests for magazine-style natural caption selection."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Keep unit assertions English-stable regardless of local UI prefs.
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
        return RawCaption(text="there is a person in a room.", source="blip", confidence=0.5)


class _ConflictingVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(
            text="One person appears to be wearing a bright red formal suit while standing outdoors.",
            source="qwen",
            confidence=0.7,
        )


def _understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
        EvidenceFact("person #1", "clothing_type", "hoodie", 0.8, "attributes"),
        EvidenceFact("person #1", "shirt_color", "navy blue", 0.8, "attributes"),
        EvidenceFact("person #1", "action", "sitting", 0.7, "pose_estimator"),
        EvidenceFact("person #1", "sitting_on", "chair", 0.75, "relationships"),
        EvidenceFact("scene", "indoor_outdoor", "indoor", 0.7, "environment"),
        EvidenceFact("scene", "setting", "office", 0.7, "environment"),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=("person #1", "chair"),
        environment_keys=("indoor_outdoor=indoor", "setting=office"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person #1: clothing_type=hoodie, shirt_color=navy blue",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _context() -> SceneContext:
    node = SceneNode(0, "obj-0", "person", 0.2, "middle-center")
    return SceneContext(
        graph=SceneGraph(nodes=(node,), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.5),
        environment=EnvironmentInfo(
            scene_type="office",
            setting="office",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="workplace",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="A person is present.",
    )


def _image() -> PreprocessedImage:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("x.jpg"),
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
    )


class _CountingVision:
    def __init__(self) -> None:
        self.narrate_calls = 0

    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        self.narrate_calls += 1
        return RawCaption(text="there is a person in a room.", source="blip", confidence=0.5)


def test_natural_caption_prefers_rich_evidence_over_thin_vlm() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _understanding(), context=_context())
    assert "navy" in paragraph.lower() or "hoodie" in paragraph.lower()
    assert "there is a person in a room" not in paragraph.lower()
    assert "person #" not in paragraph.lower()
    lower = paragraph.lower()
    for banned in (
        "is present",
        "appears to",
        "is visible",
        "objects detected",
        "confidence",
        "scene description",
    ):
        assert banned not in lower
    assert "\n" not in paragraph
    assert "objects:" not in lower
    assert "activities:" not in lower
    assert not any("\u0600" <= ch <= "\u06FF" for ch in paragraph)
    assert len(paragraph.split()) >= 10
    assert "person" in paragraph.lower()


def test_natural_caption_calls_vlm_narrate_for_image_grounding() -> None:
    vision = _CountingVision()
    service = NaturalCaptionService(vision)  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _understanding(), context=_context())
    assert vision.narrate_calls == 1
    assert len(paragraph.split()) >= 10
    assert "person" in paragraph.lower()
    # Thin VLM filler must not replace rich evidence narrative.
    assert "there is a person in a room" not in paragraph.lower()


def test_human_evidence_paragraph_reads_naturally() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    text = service._human_evidence_paragraph(_understanding())
    lower = text.lower()
    assert "hoodie" in lower or "navy" in lower
    assert "person #" not in lower
    assert lower.startswith("a person")
    assert "appears to" not in lower
    # Interaction (chair) should dominate a bolted-on clothing opener.
    first = re_split_first(text).lower()
    assert first.startswith("a person")
    assert "chair" in lower
    assert not first.startswith(("a navy", "outfit", "a hoodie"))


def test_caption_opens_with_subject_not_clothing() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _understanding(), context=_context())
    first = re_split_first(paragraph).lower()
    assert first.startswith("a person")
    assert "chair" in paragraph.lower() or "sitting" in first or "sits" in first
    assert "appears to" not in paragraph.lower()
    assert "wearing" in paragraph.lower() or "hoodie" in paragraph.lower()
    assert not first.startswith(("a navy", "outfit details", "a hoodie"))


def test_interaction_dominates_clothing_in_plan() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    scene = service._build_semantic_scene(_understanding())
    assert "sitting" in scene.defining_interaction or "chair" in scene.defining_interaction
    assert "chair" in scene.what_is_happening or "sitting" in scene.what_is_happening
    story = service._story_facts(_understanding(), scene=scene)
    assert "sitting on" in story.primary_interaction or "chair" in story.primary_interaction
    paragraph = service._compose_scene_narrative(story, scene=scene)
    first = re_split_first(paragraph).lower()
    assert "chair" in first or "sitting" in first or "sits" in first
    assert scene.story_thesis


def test_semantic_scene_built_before_language() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    scene = service._build_semantic_scene(_understanding())
    assert scene.verified_fact_count >= 4
    assert scene.attention_focus
    assert scene.what_is_happening


def test_conflicting_vlm_is_rewritten_from_evidence() -> None:
    service = NaturalCaptionService(_ConflictingVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _understanding(), context=_context())
    lower = paragraph.lower()
    assert "navy" in lower or "hoodie" in lower
    assert "red formal suit" not in lower
    assert "person #" not in lower


def test_object_scene_leads_with_object_not_person() -> None:
    facts = (
        EvidenceFact("dining table", "is", "dining table", 0.9, "yolo"),
        EvidenceFact("dining table", "dominant_color", "dark gray", 0.8, "attributes"),
        EvidenceFact("chair", "is", "chair", 0.8, "yolo"),
        EvidenceFact("scene", "indoor_outdoor", "indoor", 0.7, "environment"),
        EvidenceFact("scene", "setting", "restaurant", 0.7, "environment"),
    )
    understanding = SceneUnderstanding(
        facts=facts,
        ranked_subjects=("dining table", "chair"),
        environment_keys=("indoor_outdoor=indoor", "setting=restaurant"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    story = service._story_facts(understanding)
    assert story.scene_type in {"object-centric", "architecture", "indoor scene"}
    paragraph = service._human_evidence_paragraph(understanding)
    first = re_split_first(paragraph).lower()
    assert "person" not in first
    assert "dining table" in first or "table" in first


def test_appliance_scene_prefers_fixture_over_small_prop() -> None:
    facts = (
        EvidenceFact("book", "is", "book", 0.9, "yolo"),
        EvidenceFact("bottle", "is", "bottle", 0.9, "yolo"),
        EvidenceFact("bottle", "dominant_color", "cream", 0.8, "attributes"),
        EvidenceFact("cell phone", "is", "cell phone", 0.85, "yolo"),
        EvidenceFact("refrigerator", "is", "refrigerator", 0.85, "yolo"),
        EvidenceFact("refrigerator", "dominant_color", "light gray", 0.8, "attributes"),
        EvidenceFact("scene", "indoor_outdoor", "indoor", 0.7, "environment"),
        EvidenceFact("scene", "setting", "laboratory", 0.7, "environment"),
    )
    understanding = SceneUnderstanding(
        facts=facts,
        # Many small props ranked above the fixture — must not steal the lead.
        ranked_subjects=("book", "bottle", "cell phone", "refrigerator"),
        environment_keys=("indoor_outdoor=indoor", "setting=laboratory"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    story = service._story_facts(understanding)
    assert story.scene_type == "architecture"
    paragraph = service._human_evidence_paragraph(understanding)
    first = re_split_first(paragraph).lower()
    assert "refrigerator" in first
    assert "person" not in first
    assert not first.startswith("a cream bottle")


def test_laptop_is_object_not_document_without_ocr() -> None:
    facts = (
        EvidenceFact("laptop", "is", "laptop", 0.9, "yolo"),
        EvidenceFact("laptop", "dominant_color", "dark gray", 0.8, "attributes"),
        EvidenceFact("scene", "indoor_outdoor", "indoor", 0.7, "environment"),
        EvidenceFact("scene", "setting", "office", 0.7, "environment"),
    )
    understanding = SceneUnderstanding(
        facts=facts,
        ranked_subjects=("laptop",),
        environment_keys=("indoor_outdoor=indoor", "setting=office"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    story = service._story_facts(understanding)
    assert story.scene_type == "object-centric"
    paragraph = service._human_evidence_paragraph(understanding)
    first = re_split_first(paragraph).lower()
    assert "laptop" in first
    assert "carries the main information" not in first
    assert "person" not in first


def test_vehicle_scene_leads_with_vehicle() -> None:
    facts = (
        EvidenceFact("truck", "is", "truck", 0.9, "yolo"),
        EvidenceFact("truck", "dominant_color", "white", 0.8, "attributes"),
        EvidenceFact("car", "is", "car", 0.7, "yolo"),
        EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.7, "environment"),
        EvidenceFact("scene", "setting", "crosswalk", 0.7, "environment"),
    )
    understanding = SceneUnderstanding(
        facts=facts,
        ranked_subjects=("truck", "car"),
        environment_keys=("indoor_outdoor=outdoor", "setting=crosswalk"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    story = service._story_facts(understanding)
    assert story.scene_type == "vehicle-centric"
    paragraph = service._human_evidence_paragraph(understanding)
    first = re_split_first(paragraph).lower()
    assert first.startswith("a white truck") or first.startswith("a truck")
    assert "person" not in first


def test_competition_scorer_ranks_richer_caption_higher() -> None:
    from language.evaluation.competition_caption_scorer import CompetitionCaptionScorer

    scorer = CompetitionCaptionScorer()
    understanding = _understanding()
    context = _context()
    ranked = scorer.rank(
        [
            ("there is a person in a room.", "thin"),
            (
                "A person sits quietly. The location is an indoor, office space. "
                "Outfit details include a navy blue hoodie. The mood is calm and composed.",
                "narrative_magazine",
            ),
        ],
        context,
        understanding,
    )
    assert ranked[0].source == "narrative_magazine"


def re_split_first(text: str) -> str:
    import re

    return re.split(r"(?<=[.!?])\s+", text.strip())[0]


def test_ski_caption_is_person_led_natural_and_covers_evidence() -> None:
    """Skiing scenes must lead with the person, include jacket color + skis, no fragments."""
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.92, "yolo"),
            EvidenceFact("person #1", "clothing_color", "red", 0.84, "attributes"),
            EvidenceFact("person #1", "clothing_type", "jacket", 0.8, "attributes"),
            EvidenceFact("person #1", "jacket", "likely", 0.78, "attributes"),
            EvidenceFact("person #1", "activity", "skiing", 0.88, "activity"),
            EvidenceFact("person #1", "using", "skis", 0.82, "relationships"),
            EvidenceFact("skis", "is", "skis", 0.86, "yolo"),
            EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.8, "environment"),
            EvidenceFact("scene", "setting", "snowy mountain slope", 0.75, "environment"),
            EvidenceFact("scene", "weather", "snowy", 0.7, "environment"),
        ),
        ranked_subjects=("skis", "person #1"),  # skis ranked first — still person-led
        environment_keys=(
            "indoor_outdoor=outdoor",
            "setting=snowy mountain slope",
            "weather=snowy",
        ),
        activity_keys=("skiing",),
        ocr_text=(),
        evidence_brief="person #1 skiing; clothing_color=red; skis",
        overall_confidence=0.85,
        discarded_count=0,
        contradictions_resolved=0,
    )
    person = SceneNode(0, "obj-0", "person", 0.28, "middle-center")
    skis = SceneNode(1, "obj-1", "skis", 0.12, "bottom-center")
    context = SceneContext(
        graph=SceneGraph(nodes=(person, skis), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.8),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="snowy mountain slope",
            time_of_day="day",
            weather="snowy",
            indoor_outdoor="outdoor",
            social_context="sport",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("snow-covered slope",),
        ),
        object_count=2,
        dominant_objects=("person", "skis"),
        spatial_summary="Person skiing.",
    )
    paragraph = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), understanding, context=context
    )
    lower = paragraph.lower()
    assert "skis is" not in lower
    assert "moves through the moment" not in lower
    assert "up close, the action" not in lower
    assert any(tok in lower for tok in ("person", "skier", "someone"))
    assert "ski" in lower  # skiing and/or skis
    assert "red" in lower
    # Prefer person-led opening over equipment-led.
    first = re_split_first(paragraph).lower()
    assert not first.startswith("skis")
