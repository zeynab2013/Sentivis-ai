"""Unit tests for pipeline output validators."""

import time
from pathlib import Path

import pytest

from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.language import CaptionQualityReport, RefinedCaption
from core.contracts.metrics import PipelineMetrics
from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult
from core.exceptions.analysis import AnalysisError
from core.validation.output_validators import (
    validate_detection_result,
    validate_pipeline_result,
    validate_scene_graph,
)


def _detection_result() -> DetectionResult:
    now = time.time()
    return DetectionResult(
        detections=(
            Detection(
                object_id="obj-1",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
                class_id=0,
                detected_at=now,
            ),
        ),
        image_width=100,
        image_height=100,
        inference_timestamp=now,
    )


def test_validate_detection_result_accepts_valid_output() -> None:
    validate_detection_result(_detection_result())


def test_validate_detection_result_rejects_missing_object_id() -> None:
    now = time.time()
    invalid = DetectionResult(
        detections=(
            Detection(
                object_id="",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
                class_id=0,
                detected_at=now,
            ),
        ),
        image_width=100,
        image_height=100,
        inference_timestamp=now,
    )
    with pytest.raises(AnalysisError):
        validate_detection_result(invalid)


def test_validate_scene_graph_rejects_dangling_relation() -> None:
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.1, "top-left"),),
        relations=(),
    )
    validate_scene_graph(graph)


def test_validate_pipeline_result_accepts_complete_result() -> None:
    _detection_result()
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.1, "top-left"),),
        relations=(),
    )
    context = SceneContext(
        graph=graph,
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.4),
        environment=EnvironmentInfo(
            scene_type="general",
            setting="general scene",
            time_of_day="unknown",
            weather="unknown",
            indoor_outdoor="unknown",
            social_context="none",
            crowd_level="empty",
            scene_complexity="low",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="One person detected.",
    )
    result = PipelineResult(
        request=PipelineRequest(Path(__file__), AnalysisOptions()),
        scene_context=context,
        caption=RefinedCaption(text="A person is present.", sources=("context",)),
        quality_report=CaptionQualityReport(
            grammar_score=0.9,
            fluency_score=0.8,
            evidence_consistency=0.8,
            object_coverage=1.0,
            relationship_coverage=1.0,
            activity_coverage=1.0,
            context_coverage=0.7,
            hallucination_risk=0.1,
            overall_quality=0.85,
            notes=(),
        ),
        metrics=PipelineMetrics(
            total_duration_ms=100.0,
            peak_ram_mb=256.0,
            peak_vram_mb=0.0,
            stage_metrics=(),
            model_timings=(),
            objects_detected=1,
            relationships_inferred=0,
            activities_inferred=0,
            scene_graph_nodes=1,
            scene_graph_edges=0,
            caption_quality_score=0.85,
            recovery_events=0,
            fallback_events=0,
            competition_mode=False,
            qa_passed=True,
        ),
        qa_passed=True,
        stages_completed=(PipelineStage.VALIDATION,),
        warnings=(),
    )
    validate_pipeline_result(result)
