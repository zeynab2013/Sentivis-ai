"""Production reliability: person-role binding, bicycle/handbag activity gating."""

from __future__ import annotations

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from core.config.loader import load_analysis_config
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
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.validation.caption_factuality import (
    ClaimSupport,
    classify_sentence_against_verified,
)
from ui.formatters.result_formatters import format_activities, format_relationships
import time


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    attrs: tuple[Attribute, ...] | None = None,
    setting: str = "outdoor area",
    scene_type: str = "outdoor scene",
    indoor_outdoor: str = "outdoor",
) -> SceneContext:
    if attrs is None:
        attrs = tuple(Attribute(i, "confidence", "90%") for i in range(len(nodes)))
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=activities, confidence=0.7),
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
        dominant_objects=tuple(n.label for n in nodes[:4]),
        spatial_summary="",
    )


def _det(*items: Detection) -> DetectionResult:
    now = time.time()
    return DetectionResult(
        detections=items,
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )


def _answer(packet, question: str) -> str:
    session = VisionAssistantSession(image_key="bike", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_child_riding_bicycle_adult_walking_entity_roles() -> None:
    """person_1 rides bicycle; person_2 does not inherit riding."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.18, "middle-center"),
            SceneNode(1, "person_2", "person", 0.12, "middle-left"),
            SceneNode(2, "bicycle_1", "bicycle", 0.15, "middle-center"),
            SceneNode(3, "handbag_1", "handbag", 0.03, "middle-left"),
        ),
        relations=(
            Relation(0, 2, "riding", 0.88),
            Relation(1, 3, "carrying", 0.80),
            Relation(1, 0, "behind", 0.70),
        ),
        activities=(
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.88,
                supporting_node_indices=(0, 2),
                supporting_relation_types=("riding",),
                rationale="Person riding bicycle.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    riding = [
        r
        for r in verified.relations
        if r.relation_type == "riding" and r.narrative_safe
    ]
    assert len(riding) == 1
    assert riding[0].subject_id == "person_1"
    assert riding[0].object_id.startswith("bicycle")

    carrying = [
        r
        for r in verified.relations
        if r.relation_type == "carrying" and (r.narrative_safe or r.qa_safe)
    ]
    assert carrying
    assert carrying[0].subject_id == "person_2"

    acts = [
        a
        for a in verified.activities
        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED
    ]
    assert any("bicycle" in a.activity.lower() for a in acts)
    assert all("shopping" not in a.activity.lower() for a in verified.activities)
    assert all("driving" not in a.activity.lower() for a in verified.activities)

    summary = verified.compose_human_scene_summary().lower()
    assert "bicycle" in summary
    assert "person_1" in summary
    assert "riding" in summary

    understanding = language_understanding_from_verified(verified)
    ride_facts = [
        f
        for f in understanding.facts
        if f.predicate == "riding" or (f.predicate == "activity" and "bicycle" in f.value.lower())
    ]
    assert ride_facts
    # Activity fact binds to the rider entity, not a generic scene blob alone.
    act_facts = [f for f in understanding.facts if f.predicate == "activity"]
    assert act_facts
    assert any("person" in f.subject.lower() for f in act_facts)


def test_handbag_does_not_equal_shopping() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "handbag_1", "handbag", 0.04, "middle-center"),
        ),
        relations=(Relation(0, 1, "carrying", 0.82),),
        activities=(
            ActivityEvidence(
                activity="shopping",
                confidence=0.80,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("carrying",),
                rationale="Person carrying handbag.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity.lower() == "shopping" and a.qa_safe for a in verified.activities)
    assert any("shopping_without_cart" in r.reason for r in verified.rejected)


def test_bicycle_does_not_equal_driving() -> None:
    analyzer = HeuristicActivityAnalyzer(load_analysis_config())
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.85),),
    )
    hints = analyzer.analyze(graph)
    names = {a.activity.lower() for a in hints.activities}
    assert any("bicycle" in n or "riding" in n for n in names)
    assert "driving" not in names

    ctx = _ctx(
        graph.nodes,
        relations=graph.relations,
        activities=(
            ActivityEvidence(
                activity="driving",
                confidence=0.80,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person–vehicle interaction detected.",
            ),
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.88,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person interacting with a bicycle.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(a.activity.lower() == "driving" and a.qa_safe for a in verified.activities)
    assert any(
        a.evidence_level == ActivityEvidenceLevel.CONFIRMED and "bicycle" in a.activity.lower()
        for a in verified.activities
    )


def test_person_near_bicycle_is_not_riding() -> None:
    analyzer = HeuristicActivityAnalyzer(load_analysis_config())
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.80),),
    )
    hints = analyzer.analyze(graph)
    assert not any("riding" in a.activity.lower() for a in hints.activities)
    assert not any("cycling" in a.activity.lower() for a in hints.activities)


def test_exclusive_handbag_assignment_one_person() -> None:
    """One handbag cannot be assigned to two people as wearing+holding."""
    now = time.time()
    analyzer = RelationshipAnalyzer(load_analysis_config())
    # Overlapping people near one bag — only strongest contact should survive.
    dets = _det(
        Detection(
            object_id="p1",
            label="person",
            confidence=0.92,
            bounding_box=BoundingBox(100, 80, 220, 400),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="p2",
            label="person",
            confidence=0.90,
            bounding_box=BoundingBox(180, 90, 300, 410),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="bag",
            label="handbag",
            confidence=0.85,
            bounding_box=BoundingBox(200, 250, 250, 320),
            class_id=26,
            detected_at=now,
        ),
    )
    relations = analyzer.analyze(dets)
    possession = [
        r
        for r in relations
        if r.relation_type in {"holding", "wearing", "carrying"} and r.object_index == 2
    ]
    subjects = {r.subject_index for r in possession}
    assert len(subjects) <= 1


def test_bicycle_cannot_contain_person() -> None:
    now = time.time()
    analyzer = RelationshipAnalyzer(load_analysis_config())
    dets = _det(
        Detection(
            object_id="bike",
            label="bicycle",
            confidence=0.9,
            bounding_box=BoundingBox(100, 100, 400, 400),
            class_id=1,
            detected_at=now,
        ),
        Detection(
            object_id="person",
            label="person",
            confidence=0.92,
            bounding_box=BoundingBox(160, 140, 280, 360),
            class_id=0,
            detected_at=now,
        ),
    )
    relations = analyzer.analyze(dets)
    assert not any(
        r.relation_type == "inside"
        and (
            (r.subject_index == 1 and r.object_index == 0)
            or (r.subject_index == 0 and r.object_index == 1)
        )
        for r in relations
    )


def test_qa_uses_verified_riding_activity() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.90),),
        activities=(
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.90,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person riding bicycle.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?").lower()
    assert "bicycle" in answer or "riding" in answer
    assert "cannot determine" not in answer or "riding" in answer


def test_suggested_questions_from_verified_not_shopping() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
            SceneNode(2, "handbag_1", "handbag", 0.04, "middle-center"),
        ),
        relations=(
            Relation(0, 1, "riding", 0.90),
            Relation(0, 2, "carrying", 0.80),
        ),
        activities=(
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.90,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person riding bicycle.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(verified_evidence=verified)
    questions = generate_suggested_questions(packet)
    joined = " ".join(questions).lower()
    assert "shopping" not in joined
    assert "driving" not in joined


def test_gender_shoulder_caption_rejected_without_evidence() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "handbag_1", "handbag", 0.04, "middle-center"),
        ),
        relations=(Relation(0, 1, "carrying", 0.82),),
    )
    verified = build_verified_scene_evidence(ctx)
    bad = "The man carries a black handbag on his shoulder while the woman holds it."
    verdict = classify_sentence_against_verified(bad, verified)
    assert verdict.status == ClaimSupport.UNSUPPORTED


def test_riding_supersedes_holding_same_bicycle() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(
            Relation(0, 1, "holding", 0.90),
            Relation(0, 1, "riding", 0.88),
        ),
        activities=(
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.88,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person riding bicycle.",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    preds = {
        (r.subject_id, r.relation_type, r.object_id)
        for r in verified.relations
        if r.qa_safe or r.narrative_safe
    }
    assert any(p[1] == "riding" for p in preds)
    assert not any(p[1] == "holding" for p in preds)
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?").lower()
    assert "riding" in answer
    assert "holding" not in answer or "riding" in answer


def test_report_activities_use_verified_tiers() -> None:
    from types import SimpleNamespace

    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.90),),
        activities=(
            ActivityEvidence(
                activity="riding a bicycle",
                confidence=0.90,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="Person riding bicycle.",
            ),
            ActivityEvidence(
                activity="shopping",
                confidence=0.80,
                supporting_node_indices=(0,),
                supporting_relation_types=("carrying",),
                rationale="bag",
            ),
            ActivityEvidence(
                activity="driving",
                confidence=0.80,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("riding",),
                rationale="vehicle",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    result = SimpleNamespace(scene_context=ctx, verified_evidence=verified)
    activities_text = format_activities(result).lower()
    assert "confirmed" in activities_text
    assert "bicycle" in activities_text
    assert "shopping" not in activities_text
    assert "driving" not in activities_text
    rel_text = format_relationships(result).lower()
    assert "person" in rel_text
    assert "riding" in rel_text
