"""Unit tests for relationship containment semantics."""

import time

from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from core.config.loader import load_analysis_config
from core.contracts.detection import BoundingBox, Detection, DetectionResult


def _detections(*items: Detection) -> DetectionResult:
    now = time.time()
    return DetectionResult(
        detections=items,
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )


def test_inside_relation_orientation_and_container_gating() -> None:
    """Subject inside a larger vehicle is allowed; person inside sports ball is rejected."""
    now = time.time()
    analysis_config = load_analysis_config()
    analyzer = RelationshipAnalyzer(analysis_config)

    person_in_car = _detections(
        Detection(
            object_id="person",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(220, 180, 280, 260),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="car",
            label="car",
            confidence=0.85,
            bounding_box=BoundingBox(180, 150, 420, 320),
            class_id=2,
            detected_at=now,
        ),
    )
    car_relations = [rel for rel in analyzer.analyze(person_in_car) if rel.relation_type == "inside"]
    assert any(rel.subject_index == 0 and rel.object_index == 1 for rel in car_relations)

    person_and_ball = _detections(
        Detection(
            object_id="person",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(100, 80, 300, 400),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="ball",
            label="sports ball",
            confidence=0.8,
            bounding_box=BoundingBox(150, 200, 220, 270),
            class_id=32,
            detected_at=now,
        ),
    )
    ball_relations = [rel for rel in analyzer.analyze(person_and_ball) if rel.relation_type == "inside"]
    assert not any(
        rel.subject_index == 0 and rel.object_index == 1 for rel in ball_relations
    ), "Person must not be classified as inside a sports ball"

    two_people = _detections(
        Detection(
            object_id="p1",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(50, 50, 200, 400),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="p2",
            label="person",
            confidence=0.88,
            bounding_box=BoundingBox(120, 80, 260, 420),
            class_id=0,
            detected_at=now,
        ),
    )
    people_relations = [rel for rel in analyzer.analyze(two_people) if rel.relation_type == "inside"]
    assert not people_relations, "Person-person inside relations must be suppressed"


def test_person_horse_emits_leading_not_only_near() -> None:
    """Person beside a horse should yield an interaction predicate (leading/guiding)."""
    now = time.time()
    analysis_config = load_analysis_config()
    analyzer = RelationshipAnalyzer(analysis_config)
    scene = _detections(
        Detection(
            object_id="person",
            label="person",
            confidence=0.92,
            bounding_box=BoundingBox(80, 60, 180, 320),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="horse",
            label="horse",
            confidence=0.9,
            bounding_box=BoundingBox(150, 120, 420, 360),
            class_id=17,
            detected_at=now,
        ),
    )
    relations = analyzer.analyze(scene)
    types = {rel.relation_type for rel in relations}
    assert "leading" in types or "riding" in types or "guiding" in types
    # Interaction should suppress bare near between the same pair.
    person_horse_near = [
        rel
        for rel in relations
        if rel.relation_type == "near"
        and {rel.subject_index, rel.object_index} == {0, 1}
    ]
    assert not person_horse_near


def test_person_laptop_emits_using() -> None:
    now = time.time()
    analysis_config = load_analysis_config()
    analyzer = RelationshipAnalyzer(analysis_config)
    scene = _detections(
        Detection(
            object_id="person",
            label="person",
            confidence=0.9,
            bounding_box=BoundingBox(200, 120, 320, 360),
            class_id=0,
            detected_at=now,
        ),
        Detection(
            object_id="laptop",
            label="laptop",
            confidence=0.88,
            bounding_box=BoundingBox(220, 250, 310, 320),
            class_id=63,
            detected_at=now,
        ),
    )
    types = {rel.relation_type for rel in analyzer.analyze(scene)}
    assert "using" in types or "looking_at" in types
