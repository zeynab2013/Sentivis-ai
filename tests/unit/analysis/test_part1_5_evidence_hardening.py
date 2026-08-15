"""PART 1.5 — Evidence architecture hardening regressions."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
from analysis.relationships.relation_metrics import classify_relation, relation_kind
from core.contracts.analysis import (
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
from core.contracts.verified_evidence import ClaimStatus, RelationKind
from core.config.loader import load_analysis_config
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.refinement.caption_arbitration import (
    arbitrate_captions,
    score_caption_candidate,
)
from vision.detection.narrative_gate import is_narrative_reliable


def _env() -> EnvironmentInfo:
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


def _box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _det(index: int, label: str, box: BoundingBox, conf: float = 0.9) -> Detection:
    return Detection(
        object_id=f"{label.replace(' ', '_')}_{index}",
        label=label,
        confidence=conf,
        bounding_box=box,
        class_id=index,
        detected_at=0.0,
    )


def _ctx(
    nodes: tuple[SceneNode, ...],
    relations: tuple[Relation, ...] = (),
    attrs: tuple[Attribute, ...] = (),
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=_env(),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes),
        spatial_summary="",
    )


# --- TEST 1 ---
def test_two_people_close_no_talking_to() -> None:
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
    verified = build_verified_scene_evidence(
        _ctx(
            (
                SceneNode(0, "person_1", "person", 0.2, "middle-left"),
                SceneNode(1, "person_2", "person", 0.18, "middle-right"),
            ),
            relations=tuple(rels),
            attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
        )
    )
    assert not any(r.relation_type == "talking_to" for r in verified.relations)


# --- TEST 2 ---
def test_person_distant_phone_no_holding() -> None:
    person = _det(0, "person", _box(10, 10, 80, 200))
    phone = _det(1, "cell phone", _box(400, 300, 430, 340), conf=0.85)
    result = DetectionResult(
        detections=(person, phone),
        image_width=500,
        image_height=400,
        inference_timestamp=0.0,
    )
    rels = RelationshipAnalyzer(load_analysis_config()).analyze(result)
    assert not any(r.relation_type == "holding" for r in rels)


# --- TEST 3 ---
def test_interaction_only_when_evidence_supports() -> None:
    # Overlapping phone in hand zone can be holding; distant cannot.
    hold = classify_relation(Relation(0, 1, "holding", 0.78))
    assert hold.kind == RelationKind.INTERACTION
    assert hold.narrative_safe is True
    weak = classify_relation(Relation(0, 1, "holding", 0.40))
    assert weak.narrative_safe is False
    assert weak.qa_safe is False
    near = classify_relation(Relation(0, 1, "near", 0.85))
    assert near.kind == RelationKind.SPATIAL
    assert relation_kind("near") != relation_kind("holding")


# --- TEST 4 ---
def test_qa_shirt_color_when_caption_omits() -> None:
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.3, "middle-center"),),
        attrs=(
            Attribute(0, "confidence", "92%"),
            Attribute(0, "visibility", "high"),
            Attribute(0, "shirt_color", "navy"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = build_evidence_packet(
        verified_evidence=verified,
        canonical_caption_en="A person is sitting at a table using a laptop.",
    )
    assert packet.from_verified
    answer = VisualEvidenceRetriever().retrieve(packet, "What color is the person's shirt?")
    assert answer.direct_answer_en
    assert "navy" in answer.direct_answer_en.lower()


# --- TEST 5 ---
def test_person_name_insufficient_evidence() -> None:
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.3, "middle-center"),),
        attrs=(Attribute(0, "confidence", "90%"),),
    )
    packet = build_evidence_packet(verified_evidence=build_verified_scene_evidence(ctx))
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What is the person's name?"
    )
    assert "cannot be determined" in answer.lower()


# --- TEST 6 ---
def test_entity_identity_no_cross_contamination() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "person_2", "person", 0.18, "middle-right"),
            SceneNode(2, "phone_1", "cell phone", 0.05, "middle-left"),
            SceneNode(3, "phone_2", "cell phone", 0.04, "middle-right"),
        ),
        relations=(
            Relation(0, 2, "holding", 0.80),
            Relation(1, 3, "holding", 0.79),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "85%"),
            Attribute(3, "confidence", "84%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert verified.entity_by_index(0).entity_id == "person_1"
    assert verified.entity_by_index(1).entity_id == "person_2"
    holdings = [r for r in verified.relations if r.relation_type == "holding"]
    assert len(holdings) == 2
    pairs = {(r.subject_id, r.object_id) for r in holdings}
    assert ("person_1", "phone_1") in pairs
    assert ("person_2", "phone_2") in pairs
    assert ("person_1", "phone_2") not in pairs
    assert ("person_2", "phone_1") not in pairs
    packet = build_evidence_packet(verified_evidence=verified)
    for item in packet.items:
        if item.kind == "relation" and item.predicate == "holding":
            assert item.entity_id and item.related_entity_id
            assert item.entity_id[0] == item.related_entity_id.replace("phone", "person")[0] or True
            # person_N must map to phone_N
            assert item.entity_id.split("_")[1] == item.related_entity_id.split("_")[1]


# --- TEST 7 ---
def test_weak_detection_not_narrative_fact() -> None:
    weak = _det(0, "cell phone", _box(10, 10, 30, 40), conf=0.40)
    assert not is_narrative_reliable(weak)
    ctx = _ctx(
        (SceneNode(0, "phone_1", "cell phone", 0.02, "middle-center"),),
        attrs=(Attribute(0, "confidence", "40%"),),
    )
    verified = build_verified_scene_evidence(ctx)
    phone = verified.entity_by_id("phone_1")
    assert phone is not None
    assert phone.narrative_safe is False
    understanding = language_understanding_from_verified(verified)
    assert not any(f.value == "cell phone" for f in understanding.facts if f.predicate == "is")
    assert any(r.reason.startswith("below_narrative_floor") for r in verified.rejected)


# --- TEST 8 ---
def test_caption_candidate_unsupported_relation_rejected() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.1, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.88),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "80%")),
    )
    verified = build_verified_scene_evidence(ctx)
    score = score_caption_candidate(
        "A person is holding a chair while talking to another person.",
        verified,
    )
    assert score.rejected or score.factuality < 0.5 or score.unsupported >= 1


# --- TEST 9 ---
def test_factual_caption_beats_fluent_hallucination() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.25, "middle-right"),
            SceneNode(2, "tree_1", "tree", 0.1, "top-right"),
        ),
        relations=(Relation(0, 1, "leading", 0.82),),
        attrs=(
            Attribute(0, "confidence", "92%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "90%"),
            Attribute(2, "confidence", "80%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    factual = (
        "A person in a black shirt is leading a horse across a grassy field, "
        "with trees visible in the background."
    )
    hallucinated = (
        "A joyful teacher named Sarah is talking to her student while holding a smartphone "
        "and discussing their weekend plans near a red sports car."
    )
    winner = arbitrate_captions([hallucinated, factual], verified)
    assert "sarah" not in winner.lower()
    assert "teacher" not in winner.lower()
    assert "horse" in winner.lower() or "person" in winner.lower()


# --- TEST 10 ---
def test_detailed_caption_not_collapsed() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.22, "middle-right"),
            SceneNode(2, "horse_2", "horse", 0.15, "middle-left"),
            SceneNode(3, "tree_1", "tree", 0.08, "top-center"),
        ),
        relations=(Relation(0, 1, "leading", 0.80),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "85%"),
            Attribute(3, "confidence", "75%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    detailed = (
        "In a grassy outdoor field, a person wearing a black shirt leads a horse, "
        "while another horse stands nearby and trees rise in the background under daylight."
    )
    short = "A person is outside."
    winner = arbitrate_captions([short, detailed], verified)
    assert len(winner.split()) >= 20
    assert "horse" in winner.lower()


def test_caption_and_qa_share_same_verified_object() -> None:
    """Architectural invariant: packet and language understanding share one evidence build."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "laptop_1", "laptop", 0.1, "bottom-center"),
        ),
        relations=(
            Relation(0, 1, "using", 0.78),
            Relation(0, 1, "talking_to", 0.9),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "charcoal"),
            Attribute(1, "confidence", "85%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(r.relation_type == "talking_to" for r in verified.relations)
    packet = build_evidence_packet(verified_evidence=verified)
    understanding = language_understanding_from_verified(verified)
    assert packet.from_verified
    assert "talking" not in understanding.evidence_brief.lower()
    assert any(i.predicate == "using" for i in packet.items if i.kind == "relation")
    assert any(f.predicate == "shirt_color" for f in understanding.facts)


def test_observed_inferred_uncertain_survive() -> None:
    hold = classify_relation(Relation(0, 1, "holding", 0.78))
    near = classify_relation(Relation(0, 1, "near", 0.75))
    talk = classify_relation(Relation(0, 1, "talking_to", 0.5))
    assert hold.status == ClaimStatus.OBSERVED.value or hold.status == "OBSERVED"
    assert near.status in {"INFERRED", "OBSERVED"}
    assert talk.status == "UNCERTAIN"
