"""Integration tests for pipeline with stubbed model stages."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from certification.pipeline_stubs import StubDetector, StubReasoning, StubVisionLanguage
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from tests.support.pipeline_harness import build_test_orchestrator


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


def test_pipeline_runs_with_stub_models(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    assert result.caption.text
    assert result.scene_context.object_count == 1
    assert result.quality_report.overall_quality > 0.0
    assert result.scene_context.graph.nodes[0].object_id == "obj-test-person"
    assert result.metrics.total_duration_ms > 0.0
    assert result.metrics.objects_detected == 1


def test_pipeline_collects_metrics_in_competition_mode(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
    result = orchestrator.analyze(
        PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True, competition_mode=True))
    )
    assert result.metrics.competition_mode is True
    assert len(result.metrics.stage_metrics) >= 10
    assert result.qa_passed is True
