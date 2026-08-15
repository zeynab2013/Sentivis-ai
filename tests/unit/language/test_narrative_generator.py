"""Unit tests for narrative caption generation."""

from __future__ import annotations

import time

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import AttributeSet
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from language.semantic.narrative_generator import NarrativeGenerator


def _sample_context():
    config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="p1",
                label="person",
                confidence=0.92,
                bounding_box=BoundingBox(120, 80, 210, 260),
                class_id=0,
                detected_at=now,
            ),
            Detection(
                object_id="r1",
                label="tennis racket",
                confidence=0.88,
                bounding_box=BoundingBox(140, 120, 180, 180),
                class_id=38,
                detected_at=now,
            ),
            Detection(
                object_id="b1",
                label="sports ball",
                confidence=0.75,
                bounding_box=BoundingBox(200, 150, 220, 170),
                class_id=37,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    graph = SceneGraphBuilder(config).build(detections, ())
    activities = HeuristicActivityAnalyzer(config).analyze(graph)
    return ContextBuilder(config).build(graph, AttributeSet(attributes=()), activities)


def test_narrative_outputs_are_natural() -> None:
    context = _sample_context()
    narrative = NarrativeGenerator().generate(context)
    assert narrative.full_caption
    assert narrative.short_caption
    assert "objects include" not in narrative.full_caption.lower()
    assert len(narrative.short_caption.split()) <= 25


def test_narrative_full_meets_word_target_when_possible() -> None:
    context = _sample_context()
    narrative = NarrativeGenerator().generate(context)
    words = len(narrative.full_caption.split())
    assert words >= 30
