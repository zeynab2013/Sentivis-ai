"""End-to-end pipeline acceptance tests (headless)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from services.export.export_manager import ExportManager
from tests.acceptance.support.stubs import (
    AcceptanceStubDetector,
    AcceptanceStubReasoning,
    AcceptanceStubVisionLanguage,
)
from tests.support.pipeline_harness import build_test_orchestrator


@pytest.mark.acceptance
@pytest.mark.e2e
def test_complete_pipeline_produces_result(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))

    assert result is not None
    assert result.caption.text
    assert "person" in result.caption.text.lower()
    assert result.scene_context.object_count >= 2
    assert result.scene_context.graph.nodes
    assert result.quality_report.overall_quality > 0.0
    assert result.metrics.total_duration_ms >= 0.0


@pytest.mark.acceptance
@pytest.mark.e2e
def test_pipeline_objects_and_relationships(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))

    labels = {node.label for node in result.scene_context.graph.nodes}
    assert "person" in labels
    assert "chair" in labels
    assert result.scene_context.spatial_summary


@pytest.mark.acceptance
@pytest.mark.e2e
def test_all_export_formats(sample_image: Path, tmp_path: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    manager = ExportManager()

    paths = {
        "json": export_dir / "report.json",
        "md": export_dir / "report.md",
        "txt": export_dir / "report.txt",
        "pdf": export_dir / "report.pdf",
    }
    for fmt, path in paths.items():
        manager.export(result, fmt, path)
        assert path.is_file(), f"{fmt} export missing"
        assert path.stat().st_size > 0, f"{fmt} export empty"

    md_text = paths["md"].read_text(encoding="utf-8")
    txt_text = paths["txt"].read_text(encoding="utf-8")
    assert result.caption.text in md_text
    assert result.caption.text in txt_text
    assert paths["json"].read_text(encoding="utf-8").strip().startswith("{")
