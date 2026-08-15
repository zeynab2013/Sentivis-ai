"""Unit tests for export report builders."""

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
from services.export.export_manager import ExportManager, MarkdownExportWriter, TxtExportWriter
from services.export.report_builder import report_sections


def _sample_result() -> PipelineResult:
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
    return PipelineResult(
        request=PipelineRequest(Path("sample.png"), AnalysisOptions()),
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


def test_report_sections_include_all_required_fields() -> None:
    sections = report_sections(_sample_result())
    for key in (
        "caption",
        "scene_summary",
        "objects",
        "relationships",
        "activities",
        "context",
        "quality",
        "metrics",
    ):
        assert key in sections
        assert sections[key]


def test_markdown_and_txt_exports_write_files(tmp_path: Path) -> None:
    result = _sample_result()
    md_path = tmp_path / "report.md"
    txt_path = tmp_path / "report.txt"
    MarkdownExportWriter().write(result, md_path)
    TxtExportWriter().write(result, txt_path)
    md_text = md_path.read_text(encoding="utf-8")
    txt_text = txt_path.read_text(encoding="utf-8")
    assert "# Sentivis AI Analysis Report" in md_text
    assert "Sentivis AI Analysis Report" in txt_text
    assert "A person is present." in md_text
    assert "Caption Quality" in txt_text
    assert "Image Quality" in txt_text


def test_export_manager_supports_markdown(tmp_path: Path) -> None:
    result = _sample_result()
    path = tmp_path / "sample_sentivis.md"
    ExportManager().export(result, "md", path)
    assert path.exists()
