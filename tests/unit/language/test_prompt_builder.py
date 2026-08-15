"""Unit tests for deterministic prompt builder."""

import time
from pathlib import Path

import numpy as np

from analysis.activity.activity_analyzer import ActivityAnalyzer
from analysis.attributes.attribute_extractor import AttributeExtractor
from analysis.context.context_builder import ContextBuilder
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import SceneContext
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption, VisualObservations
from language.prompts.prompt_builder import PromptBuilder


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
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    validated = ValidatedImage(Path(__file__), 640, 480, "PNG", pixels.nbytes, pixels)
    image = PreprocessedImage(validated, pixels, pixels, 640, 480)
    attributes = AttributeExtractor(analysis_config).extract(detections, image)
    relations = RelationshipAnalyzer(analysis_config).analyze(detections)
    graph = SceneGraphBuilder(analysis_config).build(detections, relations)
    activities = ActivityAnalyzer(analysis_config).analyze(graph)
    return ContextBuilder(analysis_config).build(graph, attributes, activities)


def test_prompt_builder_is_deterministic() -> None:
    context = _context()
    observations = VisualObservations(
        observations=("A person is visible.",),
        object_attributes=("person: color=red",),
        candidate_descriptions=("A person is visible.",),
        confidence=0.8,
        raw_caption=RawCaption("A person is visible.", "blip", 0.8),
    )
    builder = PromptBuilder()
    first = builder.build(context, observations)
    second = builder.build(context, observations)
    assert first == second
    assert "SCENE GRAPH" in first.user
    assert "RELATIONSHIPS" in first.user
    assert "ACTIVITIES" in first.user
    assert "BLIP OBSERVATIONS" in first.user
    assert "Do not introduce objects" in first.system


def test_prompt_builder_without_blip_uses_context_only() -> None:
    context = _context()
    prompt = PromptBuilder().build(context, None)
    assert "Visual description unavailable" in prompt.user
