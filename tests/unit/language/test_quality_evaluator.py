"""Unit tests for caption evidence validation and quality evaluation."""

import time

from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import ActivityEvidence, ActivityHints, AttributeSet, SceneContext
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.validation.caption_validator import CaptionEvidenceValidator


def _context() -> SceneContext:
    analysis_config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="obj-person",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(10, 10, 100, 200),
                class_id=0,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    graph = SceneGraphBuilder(analysis_config).build(detections, ())
    activities = ActivityHints(
        activities=(
            ActivityEvidence(
                activity="people present",
                confidence=0.55,
                supporting_node_indices=(0,),
                supporting_relation_types=(),
                rationale="Person detected.",
            ),
        ),
        confidence=0.55,
    )
    return ContextBuilder(analysis_config).build(graph, AttributeSet(attributes=()), activities)


def test_validator_removes_unsupported_objects() -> None:
    context = _context()
    validator = CaptionEvidenceValidator()
    filtered = validator.filter_unsupported_sentences(
        "A person stands near a dragon in the scene.",
        context,
    )
    assert "dragon" not in filtered.lower()


def test_quality_evaluator_reports_metrics() -> None:
    context = _context()
    report = CaptionQualityEvaluator().evaluate(
        "A person is present in an indoor scene.",
        context,
    )
    assert 0.0 <= report.overall_quality <= 1.0
    assert report.hallucination_risk is None or 0.0 <= report.hallucination_risk <= 1.0
    assert report.object_coverage is not None and report.object_coverage >= 0.5
    # Weak placeholder activity "people present" must not invent 100% activity coverage.
    assert report.activity_coverage is None
    # No semantic relationships → N/A, not 100%.
    assert report.relationship_coverage is None


def test_activity_coverage_matches_article_variants() -> None:
    """Coverage must not report 0% when caption uses 'the' vs activity 'a'."""
    analysis_config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="obj-person",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(10, 10, 100, 200),
                class_id=0,
                detected_at=now,
            ),
            Detection(
                object_id="obj-horse",
                label="horse",
                confidence=0.9,
                bounding_box=BoundingBox(120, 10, 220, 200),
                class_id=17,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    graph = SceneGraphBuilder(analysis_config).build(detections, ())
    activities = ActivityHints(
        activities=(
            ActivityEvidence(
                activity="leading a horse",
                confidence=0.9,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("leading",),
                rationale="Verified leading.",
            ),
        ),
        confidence=0.9,
    )
    context = ContextBuilder(analysis_config).build(graph, AttributeSet(attributes=()), activities)
    report = CaptionQualityEvaluator().evaluate(
        "She is leading the horse outdoors.",
        context,
    )
    assert report.activity_coverage == 1.0
