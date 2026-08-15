"""Regression: indoor person must survive narrative + entity dedupe."""

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
        return RawCaption(text="indoor room", source="stub", confidence=0.4)


def _image() -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("indoor.jpg"),
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


def _indoor_understanding() -> SceneUnderstanding:
    return SceneUnderstanding(
        facts=(
            EvidenceFact("tv #1", "is", "tv", 0.92, "yolo"),
            EvidenceFact("tv #2", "is", "tv", 0.90, "yolo"),
            EvidenceFact("chair #1", "is", "chair", 0.84, "yolo"),
            EvidenceFact("chair #1", "dominant_color", "brown", 0.80, "attributes"),
            EvidenceFact("chair #2", "is", "chair", 0.83, "yolo"),
            EvidenceFact("vase #1", "is", "vase", 0.70, "yolo"),
            EvidenceFact("dining table #1", "is", "dining table", 0.70, "yolo"),
            EvidenceFact("refrigerator #1", "is", "refrigerator", 0.70, "yolo"),
            EvidenceFact("person #1", "is", "person", 0.70, "yolo"),
            EvidenceFact("person #1", "action", "kitchen preparation", 0.70, "pose_estimator"),
            EvidenceFact("person #1", "looking_at", "dining table", 0.60, "relationships"),
            EvidenceFact("scene", "indoor_outdoor", "indoor", 0.85, "environment"),
            EvidenceFact("scene", "setting", "kitchen", 0.80, "environment"),
        ),
        ranked_subjects=(
            "tv #1",
            "tv #2",
            "chair #1",
            "chair #2",
            "vase #1",
            "dining table #1",
            "refrigerator #1",
            "person #1",
        ),
        environment_keys=("indoor_outdoor=indoor", "setting=kitchen"),
        activity_keys=("kitchen preparation",),
        ocr_text=(),
        evidence_brief="person; kitchen preparation; chairs; refrigerator; tv; dining table",
        overall_confidence=0.75,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _indoor_context() -> SceneContext:
    nodes = (
        SceneNode(0, "obj-0", "tv", 0.05, "middle-right"),
        SceneNode(1, "obj-1", "tv", 0.02, "middle-right"),
        SceneNode(2, "obj-2", "chair", 0.02, "middle-center"),
        SceneNode(3, "obj-3", "chair", 0.02, "middle-left"),
        SceneNode(4, "obj-4", "vase", 0.01, "background"),
        SceneNode(5, "obj-5", "dining table", 0.04, "middle-center"),
        SceneNode(6, "obj-6", "refrigerator", 0.03, "middle-left"),
        SceneNode(7, "obj-7", "person", 0.02, "middle-center"),
    )
    return SceneContext(
        graph=SceneGraph(
            nodes=nodes,
            relations=(Relation(7, 5, "near", 0.7), Relation(7, 5, "looking_at", 0.6)),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(2, "dominant_color", "brown"),
                Attribute(7, "visibility", "high"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    "kitchen preparation",
                    0.78,
                    (7, 5),
                    ("holding",),
                    "person holding utensil near dining surface",
                ),
            ),
            confidence=0.7,
        ),
        environment=EnvironmentInfo(
            scene_type="kitchen",
            setting="kitchen",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="home",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("indoor kitchen",),
        ),
        object_count=8,
        dominant_objects=("tv", "chair", "person", "refrigerator", "dining table"),
        spatial_summary="Person near dining table in kitchen.",
    )


def test_indoor_person_survives_furniture_heavy_scene() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _indoor_understanding(), context=_indoor_context()
    )
    lower = caption.lower()
    assert any(tok in lower for tok in ("person", "man", "woman"))
    assert "attention stays" not in lower
    assert "softens into the wider landscape" not in lower
    assert "brown chair, and chair" not in lower
    assert lower.count("chair") <= 2
    # Prefer counted chairs over duplicated unlabeled chairs.
    assert "is close by" not in lower


def test_sanitize_drops_attention_filler() -> None:
    bad = (
        "Nearby are a refrigerator, brown chair, and chair. "
        "Behind them, a vase softens into the wider landscape. "
        "Attention stays fixed on what matters most in the foreground."
    )
    cleaned = sanitize_caption(bad)
    assert "attention stays" not in cleaned.lower()
    assert "softens into the wider landscape" not in cleaned.lower()


def test_assistant_answers_person_activity_beyond_caption() -> None:
    thin = "A refrigerator and chairs are visible in a kitchen."
    packet = build_evidence_packet(
        _indoor_context(),
        canonical_caption_en=thin,
        evidence_brief="person kitchen preparation; dining table; refrigerator",
    )
    session = VisionAssistantSession(image_key="indoor1", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What is the person doing?", language="en"
    )
    assert "caption" not in answer.lower()
    assert any(tok in answer.lower() for tok in ("prepar", "kitchen", "food", "cook"))
    assert session.assistant_vlm_calls == 0


def test_suggested_question_when_person_missing_from_caption() -> None:
    thin = "Nearby are a refrigerator and chairs."
    packet = build_evidence_packet(
        _indoor_context(),
        canonical_caption_en=thin,
        evidence_brief="person kitchen preparation",
    )
    questions = generate_suggested_questions(packet, language="en", limit=1)
    assert len(questions) <= 4
    if questions:
        assert "shoe" not in questions[0].lower()
        assert any(
            tok in questions[0].lower()
            for tok in ("person", "object", "doing", "near")
        )


def test_suggested_question_when_dining_table_omitted() -> None:
    """Borderline person (kept in graph) + omitted dining table must yield one Q."""
    dense = (
        "A person is preparing food in a kitchen. "
        "A refrigerator, a vase, and 3 chairs are visible nearby. 2 tvs are visible nearby."
    )
    # Person + fixtures with realistic confidences (person ~0.48, table ~0.61).
    nodes = (
        SceneNode(0, "obj-0", "tv", 0.05, "middle-right"),
        SceneNode(1, "obj-1", "tv", 0.02, "middle-right"),
        SceneNode(2, "obj-2", "chair", 0.02, "middle-center"),
        SceneNode(3, "obj-3", "chair", 0.02, "middle-left"),
        SceneNode(4, "obj-4", "vase", 0.01, "background"),
        SceneNode(5, "obj-5", "dining table", 0.04, "middle-center"),
        SceneNode(6, "obj-6", "refrigerator", 0.03, "middle-left"),
        SceneNode(7, "obj-7", "person", 0.02, "middle-center"),
    )
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=nodes,
            relations=(Relation(7, 5, "near", 0.7),),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "0.92"),
                Attribute(1, "confidence", "0.91"),
                Attribute(2, "confidence", "0.84"),
                Attribute(3, "confidence", "0.84"),
                Attribute(4, "confidence", "0.70"),
                Attribute(5, "confidence", "0.61"),
                Attribute(6, "confidence", "0.48"),
                Attribute(7, "confidence", "0.48"),
                Attribute(7, "visibility", "clear"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    "kitchen preparation",
                    0.78,
                    (7, 5),
                    ("holding",),
                    "person holding utensil near dining surface",
                ),
            ),
            confidence=0.7,
        ),
        environment=EnvironmentInfo(
            scene_type="kitchen",
            setting="kitchen",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="home",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("indoor kitchen",),
        ),
        object_count=8,
        dominant_objects=("tv", "chair", "person", "refrigerator", "dining table"),
        spatial_summary="Person near dining table in kitchen.",
    )
    packet = build_evidence_packet(
        ctx,
        canonical_caption_en=dense,
        evidence_brief="person; kitchen preparation; dining table omitted from caption",
    )
    questions = generate_suggested_questions(packet, language="en", limit=3)
    assert 1 <= len(questions) <= 3
    joined = " ".join(q.lower() for q in questions)
    assert "near the person" in joined or "dining table" in joined
    assert "shoe" not in joined


def test_mixed_chair_colors_do_not_force_shared_brown() -> None:
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("chair #1", "is", "chair", 0.84, "yolo"),
            EvidenceFact("chair #1", "dominant_color", "maroon", 0.80, "attributes"),
            EvidenceFact("chair #2", "is", "chair", 0.83, "yolo"),
            EvidenceFact("chair #2", "dominant_color", "brown", 0.80, "attributes"),
            EvidenceFact("chair #3", "is", "chair", 0.70, "yolo"),
            EvidenceFact("person #1", "is", "person", 0.70, "yolo"),
            EvidenceFact("scene", "setting", "kitchen", 0.80, "environment"),
        ),
        ranked_subjects=("person #1", "chair #1", "chair #2", "chair #3"),
        environment_keys=("setting=kitchen",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person; chairs",
        overall_confidence=0.7,
        discarded_count=0,
        contradictions_resolved=0,
    )
    phrases = NaturalCaptionService(_StubVision())._object_phrase_list(  # type: ignore[arg-type]
        understanding, ["chair #1", "chair #2", "chair #3"]
    )
    joined = " ".join(phrases).lower()
    assert "3 chairs" in joined
    assert "brown chairs" not in joined
    assert "maroon chairs" not in joined
