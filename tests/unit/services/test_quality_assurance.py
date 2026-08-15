"""Unit tests for post-run quality assurance."""

from core.config.app_config import CompetitionConfig
from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.language import CaptionQualityReport, RefinedCaption
from services.pipeline.quality_assurance import PipelineQualityAssurance


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


def _report(*, overall: float = 0.85, hallucination: float = 0.1) -> CaptionQualityReport:
    return CaptionQualityReport(
        grammar_score=0.9,
        fluency_score=0.8,
        evidence_consistency=0.8,
        object_coverage=1.0,
        relationship_coverage=1.0,
        activity_coverage=1.0,
        context_coverage=0.7,
        hallucination_risk=hallucination,
        overall_quality=overall,
        notes=(),
    )


def test_qa_passes_valid_caption() -> None:
    qa = PipelineQualityAssurance(
        CompetitionConfig(
            quality_threshold=0.55,
            max_hallucination_risk=0.25,
            deterministic_seed=42,
            gemma_temperature=0.0,
            vram_release_threshold_mb=64.0,
        )
    )
    result = qa.evaluate(
        RefinedCaption(text="A person is present.", sources=("gemma",)),
        _context(),
        _report(),
        strict=True,
    )
    assert result.passed is True
    assert result.rejected_caption is False


def test_qa_rejects_hallucinated_caption_in_strict_mode() -> None:
    qa = PipelineQualityAssurance(
        CompetitionConfig(
            quality_threshold=0.55,
            max_hallucination_risk=0.25,
            deterministic_seed=42,
            gemma_temperature=0.0,
            vram_release_threshold_mb=64.0,
        )
    )
    result = qa.evaluate(
        RefinedCaption(text="A helicopter and submarine are visible.", sources=("gemma",)),
        _context(),
        _report(overall=0.3, hallucination=0.75),
        strict=True,
    )
    assert result.passed is False
    assert result.rejected_caption is True
