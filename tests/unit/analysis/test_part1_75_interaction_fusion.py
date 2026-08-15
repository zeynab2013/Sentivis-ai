"""PART 1.75 — Interaction evidence fusion regressions.

Algorithmic fixtures (not real-image benchmarks).
Measures true-positive recovery vs false-positive rejection.
"""

from __future__ import annotations

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.evidence.interaction_fusion import (
    InteractionEvidenceFuser,
    extract_vlm_relation_candidates,
)
from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
from analysis.relationships.relation_metrics import classify_relation
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from core.config.loader import load_analysis_config
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
from core.contracts.language import RawCaption, VisualObservations
from core.contracts.verified_evidence import RelationKind


def _env() -> EnvironmentInfo:
    return EnvironmentInfo(
        scene_type="indoor",
        setting="room",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="indoor",
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


def _obs(text: str, conf: float = 0.8) -> VisualObservations:
    return VisualObservations(
        observations=(),
        object_attributes=(),
        candidate_descriptions=(),
        confidence=conf,
        raw_caption=RawCaption(text=text, source="test_vlm", confidence=conf),
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


def _person_phone_nodes() -> tuple[SceneNode, ...]:
    return (
        SceneNode(0, "person_1", "person", 0.25, "middle-center"),
        SceneNode(1, "phone_1", "cell phone", 0.04, "middle-center"),
    )


# --- VLM candidate parsing (not facts) ---
def test_vlm_candidate_extract_holding() -> None:
    cands = extract_vlm_relation_candidates("A person is holding a phone.", vlm_confidence=0.8)
    assert any(c.relation_type == "holding" for c in cands)


def test_vlm_alone_does_not_create_holding() -> None:
    """Case B: phone nearby + VLM says holding but no contact → reject."""
    person = _det(0, "person", _box(10, 10, 100, 220))
    phone = _det(1, "cell phone", _box(350, 280, 390, 330))
    dets = DetectionResult(
        detections=(person, phone),
        image_width=500,
        image_height=400,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        _person_phone_nodes(),
        relations=(Relation(0, 1, "near", 0.85),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("The person is holding a phone."),
    )
    holdings = [r for r in fusion.relations if r.relation_type == "holding"]
    assert holdings == []
    assert fusion.rejected_vlm >= 1


def test_vlm_plus_hand_contact_recovers_holding() -> None:
    """Case A: contact geometry + VLM → holding recovered."""
    person = _det(0, "person", _box(40, 40, 160, 260))
    # Phone overlaps torso/hand band.
    phone = _det(1, "cell phone", _box(95, 150, 130, 200))
    dets = DetectionResult(
        detections=(person, phone),
        image_width=400,
        image_height=320,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        _person_phone_nodes(),
        relations=(Relation(0, 1, "near", 0.80),),
        attrs=(Attribute(0, "confidence", "92%"), Attribute(1, "confidence", "88%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A person is holding a cell phone."),
    )
    holdings = [r for r in fusion.relations if r.relation_type == "holding"]
    assert len(holdings) == 1
    assert holdings[0].subject_index == 0
    assert holdings[0].object_index == 1
    assert fusion.recovered >= 1
    verified = build_verified_scene_evidence(
        InteractionEvidenceFuser().apply_to_context(ctx, fusion)
    )
    assert any(r.relation_type == "holding" and r.narrative_safe for r in verified.relations)


def test_distant_phone_geometry_rejects_holding() -> None:
    """Case B geometry-only: RelationshipAnalyzer must not emit holding."""
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


def test_bottle_near_not_holding() -> None:
    person = _det(0, "person", _box(20, 20, 100, 220))
    bottle = _det(1, "bottle", _box(300, 180, 330, 250))
    dets = DetectionResult(
        detections=(person, bottle),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "bottle_1", "bottle", 0.05, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.82),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "80%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A bottle sits on the table."),
    )
    assert not any(r.relation_type == "holding" for r in fusion.relations)


def test_bottle_holding_with_vlm_and_contact() -> None:
    person = _det(0, "person", _box(40, 40, 150, 250))
    bottle = _det(1, "bottle", _box(100, 140, 125, 210))
    dets = DetectionResult(
        detections=(person, bottle),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bottle_1", "bottle", 0.05, "middle-center"),
        ),
        relations=(),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("The person is holding a bottle."),
    )
    assert any(r.relation_type == "holding" for r in fusion.relations)


def test_using_laptop_recovered() -> None:
    person = _det(0, "person", _box(50, 40, 180, 260))
    laptop = _det(1, "laptop", _box(90, 180, 200, 250))
    dets = DetectionResult(
        detections=(person, laptop),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "laptop_1", "laptop", 0.08, "bottom-center"),
        ),
        relations=(Relation(0, 1, "near", 0.75),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A person is using a laptop."),
    )
    assert any(r.relation_type == "using" for r in fusion.relations)


def test_sitting_near_laptop_not_using_without_evidence() -> None:
    person = _det(0, "person", _box(20, 40, 100, 240))
    laptop = _det(1, "laptop", _box(280, 180, 360, 240))
    dets = DetectionResult(
        detections=(person, laptop),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "laptop_1", "laptop", 0.08, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.70),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A laptop is on the desk."),
    )
    assert not any(r.relation_type == "using" for r in fusion.relations)


def test_riding_bicycle_with_vlm_and_overlap() -> None:
    person = _det(0, "person", _box(80, 40, 160, 180))
    bike = _det(1, "bicycle", _box(60, 120, 200, 260))
    dets = DetectionResult(
        detections=(person, bike),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "bottom-center"),
        ),
        relations=(),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A person is riding a bicycle."),
    )
    assert any(r.relation_type == "riding" for r in fusion.relations)


def test_beside_bicycle_not_riding() -> None:
    person = _det(0, "person", _box(20, 40, 90, 220))
    bike = _det(1, "bicycle", _box(250, 120, 360, 260))
    dets = DetectionResult(
        detections=(person, bike),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.80),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("A bicycle leans against the wall."),
    )
    assert not any(r.relation_type == "riding" for r in fusion.relations)


def test_talking_to_rejected_even_with_vlm() -> None:
    """Case C: VLM says talking — still speculative without pose/gaze."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-left"),
            SceneNode(1, "person_2", "person", 0.18, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.88),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    p1 = _det(0, "person", _box(40, 40, 120, 220))
    p2 = _det(1, "person", _box(130, 45, 210, 225))
    dets = DetectionResult(
        detections=(p1, p2), image_width=400, image_height=300, inference_timestamp=0.0
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("Two people are talking to each other."),
    )
    assert not any(r.relation_type == "talking_to" for r in fusion.relations)
    verified = build_verified_scene_evidence(
        InteractionEvidenceFuser().apply_to_context(ctx, fusion)
    )
    assert not any(r.relation_type == "talking_to" for r in verified.relations)


def test_entity_grounding_person1_phone1() -> None:
    person = _det(0, "person", _box(40, 40, 150, 250))
    phone = _det(1, "cell phone", _box(95, 150, 125, 195))
    other = _det(2, "cell phone", _box(320, 200, 350, 240), conf=0.8)
    dets = DetectionResult(
        detections=(person, phone, other),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "phone_1", "cell phone", 0.04, "middle-center"),
            SceneNode(2, "phone_2", "cell phone", 0.03, "middle-right"),
        ),
        relations=(),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "80%"),
        ),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx,
        detections=dets,
        observations=_obs("The person is holding the phone."),
    )
    holdings = [r for r in fusion.relations if r.relation_type == "holding"]
    assert len(holdings) == 1
    assert holdings[0].subject_index == 0
    assert holdings[0].object_index == 1  # contact phone, not distant phone_2


def test_kitchen_person_not_automatically_preparing() -> None:
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "refrigerator_1", "refrigerator", 0.15, "middle-left"),
        ),
        relations=(Relation(0, 1, "near", 0.75),),
    )
    acts = HeuristicActivityAnalyzer(load_analysis_config()).analyze(graph)
    assert not any("prepar" in a.activity.lower() or a.activity == "cooking" for a in acts.activities)


def test_eating_requires_interaction_support() -> None:
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bowl_1", "bowl", 0.05, "middle-center"),
            SceneNode(2, "table_1", "dining table", 0.1, "middle-center"),
        ),
        relations=(Relation(0, 1, "near", 0.8), Relation(0, 2, "near", 0.7)),
    )
    acts = HeuristicActivityAnalyzer(load_analysis_config()).analyze(graph)
    assert not any(a.activity == "eating" for a in acts.activities)
    assert not any("prepar" in a.activity.lower() for a in acts.activities)


def test_spatial_vs_interaction_kind_preserved() -> None:
    near = classify_relation(Relation(0, 1, "near", 0.85))
    hold = classify_relation(Relation(0, 1, "holding", 0.78))
    assert near.kind == RelationKind.SPATIAL
    assert hold.kind == RelationKind.INTERACTION


def test_precision_recall_fixture_table() -> None:
    """Controlled fixture P/R — not a real-image benchmark.

    TP: contact+VLM holding, contact+VLM using/riding
    TN: distant+VLM holding, near-only bottle, beside bike, talking_to VLM
    """
    fuser = InteractionEvidenceFuser()
    cases: list[tuple[str, bool, bool]] = []  # name, expected_positive, predicted

    def _run(
        name: str,
        person_box: BoundingBox,
        obj_label: str,
        obj_box: BoundingBox,
        vlm: str,
        relation: str,
        expect: bool,
    ) -> None:
        person = _det(0, "person", person_box)
        obj = _det(1, obj_label, obj_box)
        dets = DetectionResult(
            detections=(person, obj),
            image_width=400,
            image_height=300,
            inference_timestamp=0.0,
        )
        ctx = _ctx(
            (
                SceneNode(0, "person_1", "person", 0.2, "middle-center"),
                SceneNode(1, f"{obj_label.replace(' ', '_')}_1", obj_label, 0.08, "middle-center"),
            ),
            relations=(Relation(0, 1, "near", 0.8),),
            attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
        )
        fusion = fuser.fuse(ctx, detections=dets, observations=_obs(vlm))
        predicted = any(r.relation_type == relation for r in fusion.relations)
        cases.append((name, expect, predicted))

    # True positives
    _run(
        "hold_phone_contact",
        _box(40, 40, 150, 250),
        "cell phone",
        _box(95, 150, 125, 195),
        "holding a phone",
        "holding",
        True,
    )
    _run(
        "use_laptop_front",
        _box(50, 40, 180, 260),
        "laptop",
        _box(90, 180, 200, 250),
        "using a laptop",
        "using",
        True,
    )
    _run(
        "ride_bike_overlap",
        _box(80, 40, 160, 180),
        "bicycle",
        _box(60, 120, 200, 260),
        "riding a bicycle",
        "riding",
        True,
    )
    # True negatives / uncertain rejected
    _run(
        "hold_phone_distant",
        _box(10, 10, 80, 200),
        "cell phone",
        _box(300, 220, 340, 270),
        "holding a phone",
        "holding",
        False,
    )
    _run(
        "bottle_near",
        _box(20, 20, 100, 220),
        "bottle",
        _box(280, 160, 310, 230),
        "a bottle on the counter",
        "holding",
        False,
    )
    _run(
        "bike_beside",
        _box(20, 40, 90, 220),
        "bicycle",
        _box(250, 120, 360, 260),
        "a bicycle nearby",
        "riding",
        False,
    )

    tp = sum(1 for _, exp, pred in cases if exp and pred)
    fn = sum(1 for _, exp, pred in cases if exp and not pred)
    fp = sum(1 for _, exp, pred in cases if not exp and pred)
    tn = sum(1 for _, exp, pred in cases if not exp and not pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    # Fixture expectations — algorithmic, not claimed real-world accuracy.
    assert tn >= 3
    assert fp == 0
    assert tp >= 2
    assert precision >= 0.99
    assert recall >= 0.66
    assert tp + tn + fp + fn == len(cases)


def test_architecture_caption_qa_still_from_verified() -> None:
    person = _det(0, "person", _box(40, 40, 150, 250))
    phone = _det(1, "cell phone", _box(95, 150, 125, 195))
    dets = DetectionResult(
        detections=(person, phone),
        image_width=400,
        image_height=300,
        inference_timestamp=0.0,
    )
    ctx = _ctx(
        _person_phone_nodes(),
        relations=(Relation(0, 1, "near", 0.8),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    fusion = InteractionEvidenceFuser().fuse(
        ctx, detections=dets, observations=_obs("holding a phone")
    )
    fused_ctx = InteractionEvidenceFuser().apply_to_context(ctx, fusion)
    verified = build_verified_scene_evidence(fused_ctx)
    from language.assistant.evidence_packet import build_evidence_packet

    packet = build_evidence_packet(verified_evidence=verified)
    assert packet.from_verified
    assert any(i.predicate == "holding" for i in packet.items if i.kind == "relation")
