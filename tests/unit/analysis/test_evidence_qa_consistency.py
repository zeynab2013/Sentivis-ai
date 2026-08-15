"""Regression: caption / evidence / QA / suggestions consistency."""

from __future__ import annotations

from analysis.context.context_builder import ContextBuilder
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
    session = VisionAssistantSession(image_key="consistency", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_people_count_matches_verified_narrative_safe() -> None:
    """QA people count must match narrative-safe entities (caption source)."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-left"),
            SceneNode(1, "person_2", "person", 0.20, "middle-right"),
            SceneNode(2, "glove_1", "baseball glove", 0.05, "middle-center"),
        ),
        attrs=(
            Attribute(0, "confidence", "92%"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "80%"),
            # Low-confidence ghost person must not affect count.
            Attribute(3, "confidence", "20%"),
        ),
        setting="outdoor scene",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    # Inject a weak third detection via an extra node with low conf.
    nodes = ctx.graph.nodes + (SceneNode(3, "person_3", "person", 0.01, "background"),)
    ctx = SceneContext(
        graph=SceneGraph(nodes=nodes, relations=ctx.graph.relations),
        attributes=ctx.attributes,
        activities=ctx.activities,
        environment=ctx.environment,
        object_count=4,
        dominant_objects=("person", "baseball glove"),
        spatial_summary="",
    )
    verified = build_verified_scene_evidence(ctx)
    narrative_people = sum(
        1
        for e in verified.entities
        if e.narrative_safe and e.label in {"person", "man", "woman", "child"}
    )
    packet = build_evidence_packet(
        verified_evidence=verified,
        canonical_caption_en="Two men are running on a baseball field.",
    )
    answer = _answer(packet, "How many people are visible?")
    assert narrative_people == 2
    assert "two" in answer.lower()
    assert "one" not in answer.lower()


def test_possession_of_racket_not_playing_tennis() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "racket_1", "tennis racket", 0.05, "middle-right"),
            SceneNode(2, "person_2", "person", 0.15, "middle-left"),
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
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(3)),
        setting="tennis court",
        scene_type="tennis court",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity == "playing tennis" and a.qa_safe for a in verified.activities)
    # Caption projection must not carry the rejected performance claim.
    understanding = language_understanding_from_verified(verified)
    activity_facts = [
        f for f in understanding.facts if f.predicate == "activity"
    ]
    assert not any(
        (f.value or "").lower().strip() == "playing tennis"
        or (f.value or "").lower().startswith("playing tennis ")
        for f in activity_facts
    )
    # Object name "tennis racket" may appear in a holding claim — that is OK.
    assert any("holding" in (f.value or "").lower() for f in activity_facts) or not activity_facts
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    lower = answer.lower()
    assert "playing tennis" not in lower
    # Possession is answerable as holding — not as performing the sport.
    assert "holding" in lower and "racket" in lower
    questions = generate_suggested_questions(packet, language="en", limit=5)
    joined = " ".join(q.lower() for q in questions)
    # Activity Q is OK only because holding is CONFIRMED; must not ask unanswerable sport Qs.
    if "what is the person doing" in joined:
        assert "holding" in VisualEvidenceRetriever().try_direct_answer(
            packet, "What is the person doing in this scene?"
        ).lower()


def test_vehicle_not_labeled_highway_from_bus_alone() -> None:
    ctx = _ctx(
        (SceneNode(0, "bus_1", "bus", 0.6, "middle-center"),),
        attrs=(Attribute(0, "confidence", "95%"),),
        setting="highway",
        scene_type="highway",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert verified.scene.scene_type != "highway"
    assert verified.scene.scene_type in {
        "urban environment",
        "outdoor scene",
        "street",
    }


def test_wildlife_gets_natural_or_outdoor_scene() -> None:
    ctx = _ctx(
        (SceneNode(0, "bear_1", "bear", 0.5, "middle-center"),),
        attrs=(Attribute(0, "confidence", "93%"),),
        setting="unknown",
        scene_type="unknown",
        indoor_outdoor="unknown",
    )
    verified = build_verified_scene_evidence(ctx)
    assert verified.scene.scene_type in {
        "natural environment",
        "outdoor scene",
    }
    assert verified.scene.scene_type != "unknown"
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    assert "playing" not in answer.lower()
    assert "riding" not in answer.lower()
    questions = generate_suggested_questions(packet, language="en", limit=5)
    assert questions  # useful suggestions even without people/activity


def test_riding_bicycle_with_riding_relation_remains_qa_safe() -> None:
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
    assert any(a.activity == "riding a bicycle" and a.qa_safe for a in verified.activities)
    understanding = language_understanding_from_verified(verified)
    assert any(
        f.predicate == "activity" and "bicycle" in (f.value or "").lower()
        for f in understanding.facts
    )
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?")
    assert "riding" in answer.lower()
    assert "driving" not in answer.lower()


def test_multi_person_count_from_verified_entities() -> None:
    nodes = tuple(
        SceneNode(i, f"person_{i+1}", "person", 0.1, "middle-center") for i in range(5)
    )
    ctx = _ctx(
        nodes,
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(5)),
        setting="outdoor scene",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "How many people are visible?")
    assert "five" in answer.lower()


def test_context_builder_bus_not_highway() -> None:
    # Call the outdoor setting helper directly — no full AnalysisConfig needed.
    scene_type, setting = ContextBuilder._specific_outdoor_setting(
        ContextBuilder.__new__(ContextBuilder),
        {"bus"},
        set(),
    )
    assert scene_type != "highway"
    assert scene_type in {"urban environment", "outdoor scene", "street"}


def test_context_builder_racket_not_auto_tennis_court() -> None:
    scene_type, setting = ContextBuilder._specific_outdoor_setting(
        ContextBuilder.__new__(ContextBuilder),
        {"person", "tennis racket"},
        set(),
    )
    assert scene_type != "tennis court"
    _ = setting


def test_suggested_questions_without_activity_still_useful() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "glove_1", "baseball glove", 0.05, "middle-right"),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "clothing_color", "white"),
            Attribute(1, "confidence", "85%"),
        ),
        setting="outdoor scene",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    packet = build_evidence_packet(
        verified_evidence=build_verified_scene_evidence(ctx),
        canonical_caption_en="A person is on a field with a glove.",
    )
    questions = generate_suggested_questions(packet, language="en", limit=5)
    assert questions
    joined = " ".join(q.lower() for q in questions)
    assert "what is the person doing" not in joined


def test_activity_answer_rejects_scene_incoherent_candidate() -> None:
    """Ranking must not return office work when scene/objects contradict it."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.25, "middle-right"),
            SceneNode(2, "tv_1", "tv", 0.02, "background"),
        ),
        relations=(Relation(0, 1, "leading", 0.88),),
        activities=(
            ActivityEvidence("office work", 0.95, (0, 2), ("using",), "spurious"),
            ActivityEvidence("leading a horse", 0.82, (0, 1), ("leading",), "strong"),
        ),
        attrs=tuple(Attribute(i, "confidence", "90%") for i in range(3)),
        setting="farm",
        scene_type="farm",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What is the person doing?"
    )
    lower = answer.lower()
    assert "office" not in lower
    assert "lead" in lower or "horse" in lower
