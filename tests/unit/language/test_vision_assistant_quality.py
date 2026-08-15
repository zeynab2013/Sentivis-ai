"""Mandatory grounding: evidence beyond caption + unknown + suggestions."""

from __future__ import annotations

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
from language.assistant.evidence_packet import build_evidence_packet, find_attribute
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.refinement.caption_sanity import sanitize_caption


def _ski_context(*, shoe_color: str | None = None, shoe_visibility: str = "low") -> SceneContext:
    person = SceneNode(0, "obj-0", "person", 0.28, "middle-center")
    skis = SceneNode(1, "obj-1", "skis", 0.12, "bottom-center")
    poles = SceneNode(2, "obj-2", "skis", 0.05, "middle-left")  # equipment proxy
    attrs = [
        Attribute(0, "confidence", "88%"),
        Attribute(0, "visibility", "high"),
        Attribute(0, "clothing_color", "red"),
        Attribute(0, "clothing_type", "jacket"),
        Attribute(0, "jacket", "likely"),
        Attribute(1, "confidence", "81%"),
        Attribute(1, "visibility", "high"),
        Attribute(2, "confidence", "75%"),
        Attribute(2, "visibility", "high"),
    ]
    if shoe_color is not None:
        attrs = [a for a in attrs if not (a.object_index == 0 and a.name == "visibility")]
        attrs.extend(
            [
                Attribute(0, "visibility", shoe_visibility),
                Attribute(0, "shoes_color", shoe_color),
            ]
        )
    return SceneContext(
        graph=SceneGraph(
            nodes=(person, skis, poles),
            relations=(Relation(0, 1, "using", 0.8),),
        ),
        attributes=AttributeSet(attributes=tuple(attrs)),
        activities=ActivityHints(
            activities=(ActivityEvidence("skiing", 0.85, (0, 1), ("using",), "ski slope"),),
            confidence=0.85,
        ),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="snowy mountain slope",
            time_of_day="day",
            weather="snowy",
            indoor_outdoor="outdoor",
            social_context="sport",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("snow-covered slope", "distant mountains"),
        ),
        object_count=3,
        dominant_objects=("person", "skis"),
        spatial_summary="Person skiing with skis on snowy slope.",
    )


def test_mandatory_wearing_answer_absent_from_caption() -> None:
    """Caption omits jacket; evidence has red jacket → assistant must answer red jacket."""
    thin_caption = "A skier is moving across a snowy slope."
    assert "jacket" not in thin_caption.lower()
    assert "red" not in thin_caption.lower()
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en=thin_caption,
        evidence_brief="person #1: clothing_color=red, clothing_type=jacket; skis",
    )
    session = VisionAssistantSession(image_key="ski-wear", evidence=packet)
    # No LLM required — direct evidence path.
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What is the skier wearing?", language="en"
    )
    assert "red" in answer.lower()
    assert "jacket" in answer.lower()
    assert "caption" not in answer.lower()
    assert "not detected" not in answer.lower()
    assert session.assistant_vlm_calls == 0


def test_mandatory_equipment_from_evidence() -> None:
    thin_caption = "A person is skiing across a snowy slope."
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en=thin_caption,
    )
    session = VisionAssistantSession(image_key="ski-eq", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What equipment is visible?", language="en"
    )
    assert "ski" in answer.lower()
    assert "not detected" not in answer.lower()
    assert session.assistant_vlm_calls == 0


def test_retriever_direct_wearing_ignores_thin_caption() -> None:
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en="A skier is moving across a snowy slope.",
    )
    result = VisualEvidenceRetriever().retrieve(packet, "What is the skier wearing?")
    assert result.direct_answer_en
    assert "red" in result.direct_answer_en.lower()
    assert "SOURCE OF TRUTH" in result.prompt_block or "VISUAL EVIDENCE FACTS" in result.prompt_block
    assert "OPTIONAL CAPTION" not in result.prompt_block


def test_unknown_shoe_color_when_weak() -> None:
    packet = build_evidence_packet(
        _ski_context(shoe_color="black", shoe_visibility="low"),
        canonical_caption_en="A skier is moving across a snowy slope.",
    )
    assert find_attribute(packet, predicate="shoes_color", require_reliable=True) is None
    session = VisionAssistantSession(image_key="ski-shoe", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What color are the shoes?", language="en"
    )
    assert "not clearly visible" in answer.lower()
    assert "black" not in answer.lower()


def test_suggestions_skip_weak_shoes_and_caption_duplicates() -> None:
    packet = build_evidence_packet(
        _ski_context(shoe_color="black", shoe_visibility="low"),
        canonical_caption_en="A person wearing a red jacket is skiing on a snowy outdoor slope.",
    )
    questions = generate_suggested_questions(packet, language="en", limit=1)
    assert 0 <= len(questions) <= 4
    joined = " ".join(q.lower() for q in questions)
    assert "shoe" not in joined
    assert "what color is the jacket" not in joined
    assert "equipment" in joined or "background" in joined or not questions


def test_sanitize_street_repetition() -> None:
    text = (
        "A person in a dark gray jacket is crossing street on a city street. "
        "A maroon car is also visible in the scene. "
        "Around them lies a city street, with trees lining the edge of the view."
    )
    cleaned = sanitize_caption(text)
    assert "crossing street on" not in cleaned.lower()
    assert "around them lies" not in cleaned.lower()
    assert cleaned.lower().count("street") <= 2
