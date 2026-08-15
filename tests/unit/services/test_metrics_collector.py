"""Unit tests for pipeline metrics collector."""

from core.config.loader import load_app_config
from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.language import CaptionQualityReport
from services.memory.memory_manager import MemoryManager
from services.pipeline.metrics_collector import PipelineMetricsCollector


def _context() -> SceneContext:
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.1, "top-left"),),
        relations=(),
    )
    return SceneContext(
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


def test_metrics_collector_records_stage_and_summary() -> None:
    memory = MemoryManager(load_app_config())
    collector = PipelineMetricsCollector(memory)
    collector.begin_run(competition_mode=True)
    collector.record_stage(PipelineStage.VALIDATION, 12.5)
    collector.record_fallback()
    report = CaptionQualityReport(
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
    )
    metrics = collector.finalize(_context(), report, qa_passed=True)
    assert metrics.competition_mode is True
    assert metrics.fallback_events == 1
    assert len(metrics.stage_metrics) == 1
    assert metrics.stage_metrics[0].stage == PipelineStage.VALIDATION
    assert metrics.objects_detected == 1
