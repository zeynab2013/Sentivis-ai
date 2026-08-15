"""Unit tests for BLIP observation mapping."""

import time

from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import ActivityHints, AttributeSet, SceneContext
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.language import RawCaption
from language.blip.observation_mapper import BlipObservationMapper


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
    return ContextBuilder(analysis_config).build(
        graph,
        AttributeSet(attributes=()),
        ActivityHints(activities=(), confidence=0.4),
    )


def test_observation_mapper_keeps_verified_attributes() -> None:
    context = _context()
    raw = RawCaption(text="A person standing in a room.", source="blip", confidence=0.85)
    mapped = BlipObservationMapper().map(raw, context)
    assert mapped.observations
    assert mapped.raw_caption.source == "blip"
    assert any("person" in item for item in mapped.candidate_descriptions)
