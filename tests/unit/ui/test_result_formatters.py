"""Unit tests for UI result formatters."""

from pathlib import Path

from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.language import CaptionQualityReport, RefinedCaption
from core.contracts.metrics import PipelineMetrics
from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult
from ui.formatters.result_formatters import (
    format_detected_objects,
    format_execution_metrics,
    format_professional_analysis,
    format_quality_report,
    format_scene_summary,
)


def test_formatters_render_pipeline_result() -> None:
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.1, "middle-center"),),
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
    assert "person" in format_scene_summary(result).lower()
    assert "person" in format_detected_objects(result).lower()
    assert "85%" in format_quality_report(result)
    assert "100.0 ms" in format_execution_metrics(result)
    professional = format_professional_analysis(result)
    assert "Scene Description:" in professional
    assert "Detected Objects:" in professional
    assert "photographed scene" not in professional.lower()
    assert "Relationships:" not in professional
    assert "Activities:" not in professional


def test_format_detected_objects_includes_vlm_fire_hazard() -> None:
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.1, "middle-center"),),
        relations=(),
    )
    context = SceneContext(
        graph=graph,
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.4),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="field",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="none",
            crowd_level="empty",
            scene_complexity="low",
            evidence=("Hazard detected: fire (confidence: 85%)",),
        ),
        object_count=1,
        dominant_objects=("fire", "person"),
        spatial_summary="Person near fire.",
    )
    result = PipelineResult(
        request=PipelineRequest(Path(__file__), AnalysisOptions()),
        scene_context=context,
        caption=RefinedCaption(text="A person is near a fire.", sources=("context",)),
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
    objects = format_detected_objects(result).lower()
    assert "person" in objects
    assert "fire" in objects
    assert "85%" in objects
