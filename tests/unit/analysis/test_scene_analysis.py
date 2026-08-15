"""Unit tests for scene analysis pipeline stages."""

import time
from pathlib import Path

import numpy as np

from analysis.activity.activity_analyzer import ActivityAnalyzer
from analysis.attributes.attribute_extractor import AttributeExtractor
from analysis.context.context_builder import ContextBuilder
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import AttributeSet
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage, ValidatedImage


def _sample_detections() -> DetectionResult:
    now = time.time()
    return DetectionResult(
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
                object_id="obj-car",
                label="car",
                confidence=0.8,
                bounding_box=BoundingBox(200, 150, 400, 300),
                class_id=2,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )


def _sample_image(detections: DetectionResult) -> PreprocessedImage:
    pixels = np.zeros((detections.image_height, detections.image_width, 3), dtype=np.uint8)
    pixels[10:200, 10:100] = (180, 40, 40)
    validated = ValidatedImage(
        path=Path(__file__),
        width=detections.image_width,
        height=detections.image_height,
        format_name="PNG",
        size_bytes=pixels.nbytes,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=validated,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=detections.image_width,
        inference_height=detections.image_height,
    )


def test_attribute_extractor_produces_rich_attributes() -> None:
    analysis_config = load_analysis_config()
    detections = _sample_detections()
    image = _sample_image(detections)
    result = AttributeExtractor(analysis_config).extract(detections, image)
    assert isinstance(result, AttributeSet)
    names = {attribute.name for attribute in result.attributes if attribute.object_index == 0}
    assert {
        "color",
        "relative_size",
        "estimated_distance",
        "pose",
        "orientation",
        "visibility",
        "occlusion",
    }.issubset(names)


def test_context_builder_creates_scene_context() -> None:
    analysis_config = load_analysis_config()
    detections = _sample_detections()
    image = _sample_image(detections)
    attributes = AttributeExtractor(analysis_config).extract(detections, image)
    relations = RelationshipAnalyzer(analysis_config).analyze(detections)
    graph = SceneGraphBuilder(analysis_config).build(detections, relations)
    activities = ActivityAnalyzer(analysis_config).analyze(graph)
    context = ContextBuilder(analysis_config).build(graph, attributes, activities)
    assert context.object_count == 2
    assert context.environment.indoor_outdoor in {"indoor", "outdoor", "unknown"}
    assert context.environment.social_context
    assert context.graph.nodes[0].object_id == "obj-person"


def test_activity_inference_requires_evidence() -> None:
    analysis_config = load_analysis_config()
    detections = _sample_detections()
    _sample_image(detections)
    relations = RelationshipAnalyzer(analysis_config).analyze(detections)
    graph = SceneGraphBuilder(analysis_config).build(detections, relations)
    activities = ActivityAnalyzer(analysis_config).analyze(graph)
    # Placeholder activities (people present / waiting) are no longer emitted.
    weak = {"people present", "static scene", "waiting", "having a conversation"}
    for item in activities.activities:
        assert item.rationale
        assert item.activity
        assert item.activity.lower() not in weak
        assert "minimal interaction" not in item.rationale.lower()
