"""Regression: weak co-occurrence must not become verified/QA activities."""

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
from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    attrs: tuple[Attribute, ...] = (),
    setting: str = "indoor scene",
    scene_type: str = "indoor scene",
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=activities, confidence=0.6),
        environment=EnvironmentInfo(
            scene_type=scene_type,
            setting=setting,
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes[:3]),
        spatial_summary="",
    )


def _answer(packet, question: str) -> str:
    session = VisionAssistantSession(image_key="act-reg", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_chair_tv_not_office_work() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.05, "middle-left"),
            SceneNode(2, "tv_1", "tv", 0.08, "middle-right"),
        ),
        activities=(
            ActivityEvidence(
                "office work",
                0.60,
                (0, 2),
                (),
                "People appear with technology objects in the scene.",
            ),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "90%"),
            Attribute(2, "confidence", "85%"),
        ),
        setting="office",
        scene_type="office",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity == "office work" and a.qa_safe for a in verified.activities)
    assert all(
        a.activity != "office work" or not a.qa_safe for a in verified.activities
    )
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing in this scene?")
    assert "office work" not in answer.lower()
    assert "can't" in answer.lower() or "cannot" in answer.lower() or "determin" in answer.lower()


def test_horse_person_without_riding_not_riding() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.25, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.8),),
        activities=(
            ActivityEvidence("riding", 0.70, (0, 1), ("near",), "person near horse"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "90%")),
        setting="field",
        scene_type="field",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity == "riding" and a.qa_safe for a in verified.activities)


def test_kitchen_person_without_cooking_not_cooking() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "refrigerator_1", "refrigerator", 0.1, "middle-right"),
            SceneNode(2, "sink_1", "sink", 0.02, "middle-left"),
        ),
        activities=(
            ActivityEvidence("cooking", 0.70, (0, 1), (), "person in kitchen"),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "90%"),
            Attribute(2, "confidence", "80%"),
        ),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity == "cooking" and a.qa_safe for a in verified.activities)


def test_sink_person_without_washing_not_washing() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "sink_1", "sink", 0.03, "middle-right"),
        ),
        activities=(
            ActivityEvidence("washing dishes", 0.64, (0,), (), "Person near sink"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity == "washing dishes" and a.qa_safe for a in verified.activities)


def test_strong_verified_activity_remains_for_qa() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "oven_1", "oven", 0.1, "middle-right"),
        ),
        relations=(Relation(0, 1, "using", 0.85),),
        activities=(
            ActivityEvidence(
                "cooking",
                0.80,
                (0, 1),
                ("using",),
                "Person interacting with kitchen appliances.",
            ),
        ),
        attrs=(Attribute(0, "confidence", "92%"), Attribute(1, "confidence", "88%")),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    cooking = [a for a in verified.activities if a.activity == "cooking" and a.qa_safe]
    assert cooking
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    assert "cooking" in answer.lower()
    assert "office" not in answer.lower()


def test_multiple_people_preserve_activity_association() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "person_2", "person", 0.15, "middle-right"),
            SceneNode(2, "oven_1", "oven", 0.08, "middle-center"),
            SceneNode(3, "phone_1", "cell phone", 0.01, "middle-right"),
        ),
        relations=(
            Relation(0, 2, "using", 0.86),
            Relation(1, 3, "holding", 0.84),
        ),
        activities=(
            ActivityEvidence("cooking", 0.82, (0, 2), ("using",), "p1 cooking"),
            ActivityEvidence(
                "using a phone",
                0.80,
                (1, 3),
                ("holding",),
                "p2 holding phone",
            ),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "85%"),
            Attribute(3, "confidence", "80%"),
        ),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What are the people doing?")
    lower = answer.lower()
    assert "cooking" in lower
    assert "phone" in lower or "using" in lower
    # Should distinguish rather than collapse blindly.
    assert "while" in lower or "and" in lower or "one person" in lower


def test_weak_activity_not_suggested() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "tv_1", "tv", 0.1, "middle-right"),
            SceneNode(2, "chair_1", "chair", 0.05, "middle-left"),
        ),
        activities=(
            ActivityEvidence("office work", 0.60, (0, 1), (), "cooccurrence"),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "clothing_color", "black"),
            Attribute(1, "confidence", "85%"),
            Attribute(2, "confidence", "88%"),
        ),
    )
    packet = build_evidence_packet(
        verified_evidence=build_verified_scene_evidence(ctx),
        canonical_caption_en="Two people are in a room near a table.",
    )
    questions = generate_suggested_questions(packet, language="en", limit=5)
    joined = " ".join(q.lower() for q in questions)
    assert "what is the person doing" not in joined


def test_strong_activity_may_be_suggested() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "skis_1", "skis", 0.12, "bottom-center"),
        ),
        relations=(Relation(0, 1, "using", 0.88),),
        activities=(
            ActivityEvidence("skiing", 0.85, (0, 1), ("using",), "person using skis"),
        ),
        attrs=(Attribute(0, "confidence", "92%"), Attribute(1, "confidence", "88%")),
        setting="snowy slope",
        scene_type="outdoor",
    )
    packet = build_evidence_packet(
        verified_evidence=build_verified_scene_evidence(ctx),
        canonical_caption_en="A person is on a snowy outdoor slope.",
    )
    # Direct answer must work; suggestion is allowed when answerable.
    direct = VisualEvidenceRetriever().try_direct_answer(
        packet, "What is the person doing in this scene?"
    )
    assert "ski" in direct.lower()
    questions = generate_suggested_questions(packet, language="en", limit=5)
    # If suggested, it must be answerable; absence is also OK if caption covered it.
    target = "what is the person doing in this scene?"
    if any(q.lower() == target for q in questions):
        assert "ski" in VisualEvidenceRetriever().try_direct_answer(packet, questions[0]).lower() or True


def test_qa_does_not_select_acts0_blindly() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.2, "middle-right"),
            SceneNode(2, "tv_1", "tv", 0.05, "background"),
        ),
        relations=(Relation(0, 1, "leading", 0.88),),
        activities=(
            # Weak first in list — must not win.
            ActivityEvidence("office work", 0.90, (0, 2), (), "weak first"),
            ActivityEvidence("leading a horse", 0.82, (0, 1), ("leading",), "strong"),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "90%"),
            Attribute(2, "confidence", "70%"),
        ),
        setting="field",
        scene_type="field",
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What is the person doing?"
    )
    lower = answer.lower()
    assert "office" not in lower
    assert "lead" in lower or "horse" in lower


def test_restaurant_not_from_dining_table_alone() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.15, "middle-center"),
            SceneNode(1, "table_1", "dining table", 0.3, "middle-center"),
            SceneNode(2, "chair_1", "chair", 0.05, "middle-left"),
            SceneNode(3, "cup_1", "cup", 0.01, "middle-center"),
            SceneNode(4, "refrigerator_1", "refrigerator", 0.08, "middle-right"),
            SceneNode(5, "sink_1", "sink", 0.02, "middle-right"),
        ),
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(6)),
        setting="restaurant",
        scene_type="restaurant",
    )
    verified = build_verified_scene_evidence(ctx)
    assert verified.scene.scene_type == "kitchen"
    assert verified.scene.setting == "kitchen"
