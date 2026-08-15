"""Regression: activity evidence levels (CONFIRMED / SUPPORTED / UNKNOWN)."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
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
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    attrs: tuple[Attribute, ...] = (),
    setting: str = "indoor scene",
    scene_type: str = "indoor scene",
    indoor_outdoor: str = "indoor",
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
            indoor_outdoor=indoor_outdoor,
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
    session = VisionAssistantSession(image_key="levels", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_kitchen_supported_preparing_food_not_unknown() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "refrigerator_1", "refrigerator", 0.1, "middle-right"),
            SceneNode(2, "sink_1", "sink", 0.05, "middle-left"),
            SceneNode(3, "bowl_1", "bowl", 0.02, "middle-center"),
        ),
        relations=(Relation(0, 3, "near", 0.75),),
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(4)),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        "office" in a.activity.lower() and a.qa_safe for a in verified.activities
    )
    supported = [
        a
        for a in verified.activities
        if a.evidence_level == ActivityEvidenceLevel.SUPPORTED
    ]
    assert supported
    packet = build_evidence_packet(
        verified_evidence=verified,
        canonical_caption_en="Two people are in a kitchen around a dining table.",
    )
    answer = _answer(packet, "What is the man doing?")
    lower = answer.lower()
    assert "can't" not in lower and "cannot" not in lower
    assert "preparing food" in lower or "kitchen" in lower
    # Caption projection must not absorb SUPPORTED-only activities.
    understanding = language_understanding_from_verified(verified)
    assert not any(
        f.predicate == "activity" and "preparing food" in (f.value or "").lower()
        for f in understanding.facts
    )


def test_horse_holding_rope_confirmed() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "rope_1", "rope", 0.05, "middle-center"),
            SceneNode(2, "horse_1", "horse", 0.25, "middle-right"),
        ),
        relations=(
            Relation(0, 1, "holding", 0.90),
            Relation(1, 2, "near", 0.80),
        ),
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(3)),
        setting="farm",
        scene_type="farm",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    confirmed = [
        a
        for a in verified.activities
        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED
        and "holding" in a.activity.lower()
        and "rope" in a.activity.lower()
    ]
    assert confirmed
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    lower = answer.lower()
    assert "holding" in lower and "rope" in lower
    assert "riding" not in lower


def test_bicycle_riding_remains_confirmed() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "bike_1", "bicycle", 0.2, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.90),),
        activities=(
            ActivityEvidence(
                "riding a bicycle",
                0.90,
                (0, 1),
                ("riding",),
                "person riding bicycle",
            ),
        ),
        attrs=(Attribute(0, "confidence", "92%"), Attribute(1, "confidence", "90%")),
        setting="natural environment",
        scene_type="natural environment",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert any(
        a.evidence_level == ActivityEvidenceLevel.CONFIRMED
        and "bicycle" in a.activity.lower()
        for a in verified.activities
    )
    answer = _answer(build_evidence_packet(verified_evidence=verified), "What is the person doing?")
    assert "riding" in answer.lower()
    assert "driving" not in answer.lower()


def test_false_office_still_rejected() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.05, "middle-left"),
            SceneNode(2, "tv_1", "tv", 0.08, "middle-right"),
            SceneNode(3, "phone_1", "cell phone", 0.01, "middle-center"),
        ),
        activities=(
            ActivityEvidence(
                "office work",
                0.60,
                (0, 1, 2),
                (),
                "People appear with technology objects in the scene.",
            ),
        ),
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(4)),
        setting="office",
        scene_type="office",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        "office" in a.activity.lower()
        and a.evidence_level
        in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        for a in verified.activities
    )
    answer = _answer(build_evidence_packet(verified_evidence=verified), "What is the person doing?")
    assert "office" not in answer.lower()


def test_tennis_holding_racket_not_playing() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "racket_1", "tennis racket", 0.05, "middle-right"),
        ),
        relations=(Relation(0, 1, "holding", 0.88),),
        activities=(
            ActivityEvidence(
                "playing tennis",
                0.90,
                (0, 1),
                ("holding",),
                "person holding racket",
            ),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
        setting="outdoor scene",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        "playing tennis" in a.activity.lower()
        and a.evidence_level
        in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        for a in verified.activities
    )
    answer = _answer(build_evidence_packet(verified_evidence=verified), "What is the person doing?")
    lower = answer.lower()
    assert "playing tennis" not in lower
    assert "holding" in lower and "racket" in lower


def test_phone_strong_interaction_supported_or_confirmed() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "phone_1", "cell phone", 0.02, "middle-center"),
        ),
        relations=(Relation(0, 1, "looking_at", 0.86),),
        activities=(
            ActivityEvidence(
                "using phone",
                0.78,
                (0, 1),
                ("looking_at",),
                "person looking at phone near face",
            ),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    verified = build_verified_scene_evidence(ctx)
    answerable = [
        a
        for a in verified.activities
        if a.evidence_level
        in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        and ("phone" in a.activity.lower() or "looking" in a.activity.lower())
    ]
    assert answerable
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    lower = answer.lower()
    assert "can't" not in lower
    assert "phone" in lower or "looking" in lower
    questions = generate_suggested_questions(packet, language="en", limit=5)
    # Activity suggestion allowed when answerable evidence exists.
    assert questions
