"""Regression tests: evidence-first Vision Assistant QA routing."""

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
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.entity_indexing import ordered_people, resolve_person_reference
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


def _kitchen_two_people_context(
    *,
    person2_clothing: str | None = "blue",
    person2_clothing_conf_attr: bool = True,
    include_ocr: bool = False,
) -> SceneContext:
    _ = include_ocr
    nodes = (
        SceneNode(0, "person_1", "person", 0.09, "middle-left"),
        SceneNode(1, "person_2", "person", 0.04, "middle-right"),
        SceneNode(2, "dining_table_1", "dining table", 0.30, "middle-center"),
        SceneNode(3, "chair_1", "chair", 0.03, "middle-left"),
        SceneNode(4, "chair_2", "chair", 0.03, "middle-right"),
        SceneNode(5, "chair_3", "chair", 0.03, "bottom-left"),
        SceneNode(6, "refrigerator_1", "refrigerator", 0.06, "middle-right"),
        SceneNode(7, "car_1", "car", 0.05, "background"),
    )
    attrs = [
        Attribute(0, "confidence", "90%"),
        Attribute(0, "visibility", "high"),
        Attribute(0, "clothing_color", "black"),
        Attribute(0, "shirt_color", "black"),
        Attribute(0, "clothing_type", "jacket"),
        Attribute(0, "jacket", "likely"),
        Attribute(1, "confidence", "85%"),
        Attribute(1, "visibility", "high"),
        Attribute(2, "confidence", "90%"),
        Attribute(2, "visibility", "high"),
        Attribute(3, "confidence", "88%"),
        Attribute(3, "visibility", "high"),
        Attribute(3, "dominant_color", "brown"),
        Attribute(3, "color", "brown"),
        Attribute(4, "confidence", "87%"),
        Attribute(4, "visibility", "high"),
        Attribute(5, "confidence", "86%"),
        Attribute(5, "visibility", "high"),
        Attribute(6, "confidence", "91%"),
        Attribute(6, "visibility", "high"),
        Attribute(7, "confidence", "80%"),
        Attribute(7, "visibility", "high"),
        Attribute(7, "dominant_color", "red"),
        Attribute(7, "color", "red"),
    ]
    if person2_clothing is not None and person2_clothing_conf_attr:
        attrs.extend(
            [
                Attribute(1, "clothing_color", person2_clothing),
                Attribute(1, "shirt_color", person2_clothing),
            ]
        )
    return SceneContext(
        graph=SceneGraph(
            nodes=nodes,
            relations=(
                Relation(0, 2, "near", 0.8),
                Relation(1, 2, "near", 0.75),
                Relation(6, 2, "behind", 0.7),
            ),
        ),
        attributes=AttributeSet(attributes=tuple(attrs)),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    "standing",
                    0.6,
                    (0, 1, 2),
                    ("near",),
                    "people near dining table",
                ),
            ),
            confidence=0.6,
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
        object_count=8,
        dominant_objects=("person", "dining table", "chair", "refrigerator"),
        spatial_summary="Two people near a dining table in a kitchen.",
    )


def _packet(**kwargs):  # type: ignore[no-untyped-def]
    ctx = _kitchen_two_people_context(**kwargs)
    return build_evidence_packet(
        ctx,
        canonical_caption_en=(
            "Two people are in a kitchen around a dining table. "
            "A refrigerator is visible behind them."
        ),
        evidence_brief="two people; dining table; chairs; refrigerator; kitchen",
        ocr_snippets=("OPEN",) if kwargs.get("include_ocr") else (),
    )


def _answer(packet, question: str) -> str:
    session = VisionAssistantSession(image_key="qa-reg", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_person_count_from_verified_evidence() -> None:
    answer = _answer(_packet(), "How many people are visible?")
    assert "two" in answer.lower()
    assert "people" in answer.lower()


def test_object_count_from_verified_evidence() -> None:
    answer = _answer(_packet(), "How many chairs are there?")
    assert "three" in answer.lower()
    assert "chair" in answer.lower()


def test_object_color_from_verified_evidence() -> None:
    answer = _answer(_packet(), "What color is the car?")
    assert "red" in answer.lower()


def test_second_person_clothing_color_when_verified() -> None:
    packet = _packet(person2_clothing="blue")
    people = ordered_people(packet)
    assert len(people) >= 2
    second = resolve_person_reference(
        "What color clothing is the second person wearing?", packet
    )
    assert second is not None
    assert second.ordinal == 2
    answer = _answer(packet, "What color clothing is the second person wearing?")
    lower = answer.lower()
    assert "blue" in lower
    assert "second person" in lower
    assert "can't reliably" not in lower


def test_indexed_person_uses_matching_entity() -> None:
    packet = _packet(person2_clothing="blue")
    first = _answer(packet, "What color clothing is the first person wearing?")
    second = _answer(packet, "What color clothing is the second person wearing?")
    assert "black" in first.lower()
    assert "blue" in second.lower()
    assert first.lower() != second.lower()


def test_spatial_where_is_refrigerator() -> None:
    answer = _answer(_packet(), "Where is the refrigerator?")
    lower = answer.lower()
    assert "refrigerator" in lower
    assert any(tok in lower for tok in ("behind", "near", "visible", "right", "scene"))


def test_presence_question() -> None:
    yes = _answer(_packet(), "Is there a horse?")
    # No horse in context — honest negative or uncertainty, never invent yes.
    assert not (yes.lower().startswith("yes") and "horse" in yes.lower())
    fridge = _answer(_packet(), "Is there a refrigerator?")
    assert "yes" in fridge.lower()
    assert "refrigerator" in fridge.lower()


def test_ocr_question() -> None:
    answer = _answer(_packet(include_ocr=True), "What does the sign say?")
    assert "open" in answer.lower()


def test_unavailable_clothing_does_not_hallucinate() -> None:
    packet = _packet(person2_clothing=None)
    answer = _answer(packet, "What color clothing is the second person wearing?")
    lower = answer.lower()
    assert "blue" not in lower
    assert "red" not in lower
    assert "can't reliably" in lower or "cannot" in lower or "can't match" in lower


def test_suggested_questions_only_when_answerable() -> None:
    # With verified second-person blue clothing, the suggestion may appear.
    with_color = _packet(person2_clothing="blue")
    qs_yes = generate_suggested_questions(with_color, language="en", limit=5)
    # Without second-person clothing evidence, do not suggest that question.
    no_color = _packet(person2_clothing=None)
    qs_no = generate_suggested_questions(no_color, language="en", limit=5)
    joined_no = " ".join(qs_no).lower()
    assert "what color clothing is the second person wearing?" not in joined_no
    # If suggested with color evidence, the retriever must answer it.
    target = "What color clothing is the second person wearing?"
    if any(q.lower() == target.lower() for q in qs_yes):
        direct = VisualEvidenceRetriever().try_direct_answer(with_color, target)
        assert "blue" in direct.lower()
        assert "can't reliably" not in direct.lower()


def test_general_visual_falls_through_when_no_direct_fact() -> None:
    """Open questions may lack a direct template; retrieval still has evidence."""
    packet = _packet()
    result = VisualEvidenceRetriever().retrieve(packet, "What is happening in this scene?")
    # Either a direct activity answer or selected evidence for LLM fusion.
    assert result.direct_answer_en or result.selected or result.prompt_block
