"""PART 1 visual intelligence regressions — relations, evidence gating, QA grounding."""

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
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.config.loader import load_analysis_config
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from analysis.relationships.relation_metrics import (
    RelationEvidenceTier,
    caption_safe_relations,
    classify_relation,
    meaningful_relations,
    qa_safe_relations,
)
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.validation.caption_factuality import ClaimSupport, classify_sentence
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from vision.detection.narrative_gate import filter_narrative_detections, is_narrative_reliable


def _box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _det(index: int, label: str, box: BoundingBox, conf: float = 0.9) -> Detection:
    return Detection(
        object_id=f"obj_{index}",
        label=label,
        confidence=conf,
        bounding_box=box,
        class_id=index,
        detected_at=0.0,
    )


def _empty_env() -> EnvironmentInfo:
    return EnvironmentInfo(
        scene_type="outdoor",
        setting="field",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="outdoor",
        social_context="",
        crowd_level="few",
        scene_complexity="medium",
        evidence=(),
    )


def test_holding_requires_overlap_not_mere_proximity() -> None:
    """Phone across the frame must not become holding."""
    person = _det(0, "person", _box(10, 10, 80, 200))
    phone = _det(1, "cell phone", _box(400, 300, 430, 340), conf=0.85)
    result = DetectionResult(
        detections=(person, phone),
        image_width=500,
        image_height=400,
        inference_timestamp=0.0,
    )
    rels = RelationshipAnalyzer(load_analysis_config()).analyze(result)
    holdings = [r for r in rels if r.relation_type == "holding"]
    assert holdings == []


def test_talking_to_not_emitted_from_proximity() -> None:
    """Two close people must not invent talking_to."""
    p1 = _det(0, "person", _box(40, 40, 120, 220))
    p2 = _det(1, "person", _box(130, 45, 210, 225))
    result = DetectionResult(
        detections=(p1, p2),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    rels = RelationshipAnalyzer(load_analysis_config()).analyze(result)
    assert not any(r.relation_type == "talking_to" for r in rels)
    assert not any(r.relation_type == "looking_at" for r in rels)


def test_looking_at_not_emitted_for_nearby_book() -> None:
    person = _det(0, "person", _box(20, 20, 100, 200))
    book = _det(1, "book", _box(110, 120, 160, 180))
    result = DetectionResult(
        detections=(person, book),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    rels = RelationshipAnalyzer(load_analysis_config()).analyze(result)
    assert not any(r.relation_type == "looking_at" for r in rels)


def test_relation_tiers_exclude_weak_spatial_from_caption() -> None:
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bottle_1", "bottle", 0.05, "middle-right"),
        ),
        relations=(
            Relation(0, 1, "near", 0.80),
            Relation(0, 1, "holding", 0.75),
            Relation(0, 1, "talking_to", 0.70),
        ),
    )
    near = classify_relation(graph.relations[0])
    hold = classify_relation(graph.relations[1])
    talk = classify_relation(graph.relations[2])
    # Spatial near may be QA-safe as SPATIAL — never as interaction / narrative action.
    assert near.kind.value == "SPATIAL"
    assert near.narrative_safe is False
    assert near.qa_safe is True
    assert hold.tier == RelationEvidenceTier.HIGH
    assert hold.kind.value == "INTERACTION"
    assert talk.status == "UNCERTAIN"
    safe = caption_safe_relations(graph)
    assert len(safe) == 1
    assert safe[0].relation_type == "holding"
    qa = qa_safe_relations(graph)
    assert any(r.relation_type == "near" for r in qa)
    assert all(r.relation_type != "talking_to" for r in qa)
    meaningful = meaningful_relations(graph)
    assert any(r.relation_type == "holding" for r in meaningful)
    # High-confidence spatial near is retained as layout evidence (not interaction).
    assert any(r.relation_type == "near" for r in meaningful)


def test_evidence_packet_spatial_near_not_holding() -> None:
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.12, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.88),),
    )
    ctx = SceneContext(
        graph=graph,
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(1, "confidence", "80%"),
            )
        ),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=_empty_env(),
        object_count=2,
        dominant_objects=("person", "chair"),
        spatial_summary="",
    )
    packet = build_evidence_packet(ctx, canonical_caption_en="A person stands outdoors.")
    assert packet.from_verified is True
    near_items = [i for i in packet.items if i.kind == "relation" and i.predicate == "near"]
    assert near_items
    assert near_items[0].relation_kind == "SPATIAL"
    assert near_items[0].entity_id == "person_1"
    assert near_items[0].related_entity_id == "chair_1"
    retriever = VisualEvidenceRetriever()
    answer = retriever._near_objects_answer(packet, "what is near the person")
    assert "near" in answer.lower()
    assert "holding" not in answer.lower()
    assert "interacting" not in answer.lower()


def test_qa_answers_color_absent_from_caption() -> None:
    graph = SceneGraph(
        nodes=(SceneNode(0, "person_1", "person", 0.3, "middle-center"),),
        relations=(),
    )
    ctx = SceneContext(
        graph=graph,
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "92%"),
                Attribute(0, "visibility", "high"),
                Attribute(0, "shirt_color", "navy"),
            )
        ),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=_empty_env(),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="",
    )
    packet = build_evidence_packet(
        ctx,
        canonical_caption_en="A person is sitting at a table using a laptop.",
    )
    retriever = VisualEvidenceRetriever()
    direct = retriever.retrieve(packet, "What color is the person's shirt?")
    assert direct.direct_answer_en
    assert "navy" in direct.direct_answer_en.lower()


def test_factuality_rejects_unsupported_talking() -> None:
    understanding = SceneUnderstanding(
        facts=(EvidenceFact("person #1", "is", "person", 0.9, "yolo"),),
        ranked_subjects=("person #1",),
        environment_keys=(),
        activity_keys=(),
        ocr_text=(),
        overall_confidence=0.8,
        evidence_brief="Objects\n- person",
        discarded_count=0,
        contradictions_resolved=0,
    )
    verdict = classify_sentence(
        "A person is talking to another person.",
        understanding,
    )
    assert verdict.status == ClaimSupport.UNSUPPORTED


def test_narrative_gate_keeps_weak_detection_out_of_language() -> None:
    weak_phone = _det(0, "cell phone", _box(10, 10, 30, 40), conf=0.40)
    strong_person = _det(1, "person", _box(50, 20, 120, 200), conf=0.85)
    assert not is_narrative_reliable(weak_phone)
    kept = filter_narrative_detections((weak_phone, strong_person))
    assert len(kept) == 1
    assert kept[0].label == "person"


def test_multi_person_entities_keep_distinct_object_ids() -> None:
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "person_2", "person", 0.18, "middle-right"),
            SceneNode(2, "laptop_1", "laptop", 0.08, "bottom-center"),
        ),
        relations=(
            Relation(0, 2, "using", 0.78),
            Relation(1, 2, "near", 0.70),
        ),
    )
    assert graph.nodes[0].object_id != graph.nodes[1].object_id
    safe = caption_safe_relations(graph)
    assert len(safe) == 1
    assert safe[0].subject_index == 0
    assert safe[0].object_index == 2
