"""Object-lexicon QA: no false hallucinations on ordinary kitchen language."""

from __future__ import annotations

import time

from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import ActivityEvidence, ActivityHints, AttributeSet
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from language.validation.caption_validator import CaptionEvidenceValidator


def _kitchen_context():
    analysis_config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="obj-person",
                label="person",
                confidence=0.92,
                bounding_box=BoundingBox(40, 40, 220, 400),
                class_id=0,
                detected_at=now,
            ),
            Detection(
                object_id="obj-bowl",
                label="bowl",
                confidence=0.88,
                bounding_box=BoundingBox(260, 280, 340, 340),
                class_id=45,
                detected_at=now,
            ),
            Detection(
                object_id="obj-sink",
                label="sink",
                confidence=0.85,
                bounding_box=BoundingBox(360, 260, 520, 360),
                class_id=71,
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
                activity="preparing food",
                confidence=0.7,
                supporting_node_indices=(0, 1),
                supporting_relation_types=(),
                rationale="Person near bowl and sink.",
            ),
        ),
        confidence=0.7,
    )
    context = ContextBuilder(analysis_config).build(graph, AttributeSet(attributes=()), activities)
    # Force kitchen environment wording into evidence when builder is generic.
    if "kitchen" not in context.environment.scene_type.lower():
        from dataclasses import replace

        context = replace(
            context,
            environment=replace(
                context.environment,
                scene_type="kitchen",
                setting="kitchen",
                indoor_outdoor="indoor",
            ),
        )
    return context


def test_kitchen_phrases_are_not_false_hallucinations() -> None:
    context = _kitchen_context()
    validator = CaptionEvidenceValidator()
    caption = (
        "A person is preparing food near a bowl and sink in a kitchen."
    )
    unsupported = validator.unsupported_object_tokens(caption, context)
    for token in ("person", "bowl", "sink", "kitchen", "food"):
        assert token not in unsupported, unsupported
    assert "preparing" not in unsupported


def test_ordinary_english_words_not_object_hallucinations() -> None:
    context = _kitchen_context()
    validator = CaptionEvidenceValidator()
    caption = "A compact person stays focused while preparing food by the sink."
    unsupported = validator.unsupported_object_tokens(caption, context)
    assert "compact" not in unsupported
    assert "stay" not in unsupported
    assert "stays" not in unsupported
    assert "focused" not in unsupported


def test_unsupported_concrete_object_still_flags() -> None:
    context = _kitchen_context()
    validator = CaptionEvidenceValidator()
    unsupported = validator.unsupported_object_tokens(
        "A person stands near a dragon in the kitchen.",
        context,
    )
    assert "dragon" in unsupported
