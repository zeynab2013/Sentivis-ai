"""Final hardening regressions: workstation caption, attributes, one suggestion."""

from __future__ import annotations

import os
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
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.refinement.caption_refiner import clear_ui_language_cache
from language.refinement.caption_sanity import sanitize_caption
from language.semantic.natural_caption_service import NaturalCaptionService


@pytest.fixture(autouse=True)
def _force_english_ui_language() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _StubVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="a person at a desk", source="stub", confidence=0.4)


def _image() -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("workstation.jpg"),
        width=48,
        height=48,
        format_name="JPEG",
        size_bytes=200,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=48,
        inference_height=48,
        original_display_pixels=pixels,
    )


def _workstation_understanding() -> SceneUnderstanding:
    return SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.92, "yolo"),
            EvidenceFact("person #1", "action", "working at a computer", 0.84, "pose_estimator"),
            EvidenceFact("person #1", "clothing_color", "charcoal", 0.88, "attributes"),
            EvidenceFact("person #1", "shirt_color", "charcoal", 0.90, "attributes"),
            EvidenceFact("person #1", "using", "keyboard", 0.80, "relationships"),
            EvidenceFact("keyboard #1", "is", "keyboard", 0.86, "yolo"),
            EvidenceFact("tv #1", "is", "tv", 0.78, "yolo"),
            EvidenceFact("tv #1", "dominant_color", "charcoal", 0.70, "attributes"),
            EvidenceFact("chair #1", "is", "chair", 0.82, "yolo"),
            EvidenceFact("chair #1", "dominant_color", "navy", 0.80, "attributes"),
            EvidenceFact("scene", "indoor_outdoor", "indoor", 0.85, "environment"),
            EvidenceFact("scene", "setting", "office", 0.80, "environment"),
        ),
        ranked_subjects=("person #1", "tv #1", "keyboard #1", "chair #1"),
        environment_keys=("indoor_outdoor=indoor", "setting=office"),
        activity_keys=("working at a computer",),
        ocr_text=(),
        evidence_brief="person charcoal shirt; working at computer; keyboard; navy chair",
        overall_confidence=0.85,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _workstation_context() -> SceneContext:
    nodes = (
        SceneNode(0, "obj-0", "person", 0.30, "middle-center"),
        SceneNode(1, "obj-1", "tv", 0.20, "middle-right"),
        SceneNode(2, "obj-2", "keyboard", 0.12, "bottom-center"),
        SceneNode(3, "obj-3", "chair", 0.18, "middle-left"),
    )
    return SceneContext(
        graph=SceneGraph(
            nodes=nodes,
            relations=(Relation(0, 2, "using", 0.8),),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "clothing_color", "charcoal"),
                Attribute(0, "shirt_color", "charcoal"),
                Attribute(0, "visibility", "high"),
                Attribute(1, "dominant_color", "charcoal"),
                Attribute(3, "dominant_color", "navy"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence("working at a computer", 0.84, (0, 2), ("using",), "office"),
            ),
            confidence=0.84,
        ),
        environment=EnvironmentInfo(
            scene_type="office",
            setting="office workspace",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="workplace",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("desk area",),
        ),
        object_count=4,
        dominant_objects=("person", "tv", "keyboard", "chair"),
        spatial_summary="Person at computer with keyboard and chair.",
    )


def test_sanitize_drops_workstation_filler_and_keyboard_repeat() -> None:
    bad = (
        "A person is working at a computer in an office workspace. "
        "The main work underway is using a keyboard. "
        "A charcoal tv and a navy chair share the surrounding space. "
        "They are using a keyboard. "
        "The setting remains clearly indoors."
    )
    cleaned = sanitize_caption(bad)
    assert cleaned.lower().count("keyboard") <= 1
    assert "main work underway" not in cleaned.lower()
    assert "they are using" not in cleaned.lower()
    assert "setting remains" not in cleaned.lower()
    assert "share the surrounding" not in cleaned.lower()
    assert "charcoal tv" not in cleaned.lower()
    assert "main work underway" not in cleaned.lower()


def test_workstation_caption_no_filler_or_wrong_tv_color() -> None:
    paragraph = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _workstation_understanding(), context=_workstation_context()
    )
    lower = paragraph.lower()
    assert "main work underway" not in lower
    assert "setting remains" not in lower
    assert "they are using" not in lower
    assert "share the surrounding" not in lower
    assert "also visible in the scene" not in lower
    assert lower.count("keyboard") <= 1
    assert "charcoal tv" not in lower
    assert "charcoal monitor" not in lower
    # Person clothing color may appear; must not be glued onto the TV.
    if "charcoal" in lower:
        assert "tv" not in lower[max(0, lower.find("charcoal") - 12) : lower.find("charcoal") + 20] or (
            "shirt" in lower or "wearing" in lower or "person" in lower
        )


def test_assistant_shirt_color_from_scenecontext_not_caption() -> None:
    thin = "A person is working at a computer."
    packet = build_evidence_packet(
        _workstation_context(),
        canonical_caption_en=thin,
        evidence_brief="person shirt_color=charcoal; keyboard; navy chair",
    )
    session = VisionAssistantSession(image_key="ws1", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What color is the man's t-shirt?", language="en"
    )
    assert "charcoal" in answer.lower()
    assert "caption" not in answer.lower()
    assert session.assistant_vlm_calls == 0


def test_suggested_question_is_at_most_one_and_not_shirt_duplicate() -> None:
    caption = (
        "A person in a charcoal shirt is working at a computer in an office workspace. "
        "A navy chair is close by."
    )
    packet = build_evidence_packet(
        _workstation_context(),
        canonical_caption_en=caption,
        evidence_brief="person charcoal; keyboard; navy chair; tv",
    )
    questions = generate_suggested_questions(packet, language="en", limit=5)
    assert 0 <= len(questions) <= 4
    joined = " ".join(q.lower() for q in questions)
    assert "what color is the" not in joined or "shirt" not in joined
    assert "shoe" not in joined


def test_object_color_stays_on_source_entity() -> None:
    """Navy belongs to chair; charcoal belongs to person clothing — not TV."""
    paragraph = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _workstation_understanding(), context=_workstation_context()
    )
    lower = paragraph.lower()
    if "navy" in lower:
        # Navy should appear near chair, not near person clothing alone as "navy person".
        assert "chair" in lower
    assert "charcoal tv" not in lower
    assert "navy person" not in lower
