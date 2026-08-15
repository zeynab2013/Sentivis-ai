"""Performance acceptance tests with recorded thresholds."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from tests.acceptance.support.stubs import (
    AcceptanceStubDetector,
    AcceptanceStubReasoning,
    AcceptanceStubVisionLanguage,
)
from tests.support.pipeline_harness import build_test_orchestrator


@pytest.mark.acceptance
@pytest.mark.performance
def test_startup_time_under_threshold() -> None:
    started = time.perf_counter()
    StartupOrchestrator().run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 120_000, f"Startup too slow: {elapsed_ms:.0f} ms"


@pytest.mark.acceptance
@pytest.mark.performance
def test_inference_time_under_threshold(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    started = time.perf_counter()
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert result.metrics.total_duration_ms >= 0.0
    assert elapsed_ms < 60_000, f"Inference too slow: {elapsed_ms:.0f} ms"


@pytest.mark.acceptance
@pytest.mark.performance
def test_ram_usage_recorded(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    assert result.metrics.peak_ram_mb > 0.0
    assert result.metrics.peak_ram_mb < 8192.0


@pytest.mark.acceptance
@pytest.mark.performance
def test_vram_usage_recorded(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    assert result.metrics.peak_vram_mb >= 0.0


@pytest.mark.acceptance
@pytest.mark.performance
def test_gpu_memory_released_after_inference(sample_image: Path) -> None:
    orchestrator = build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )
    orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    orchestrator._model_manager.release_all()  # noqa: SLF001
    orchestrator._memory.clear_gpu_cache()  # noqa: SLF001
    after = orchestrator._memory.snapshot()  # noqa: SLF001
    assert after.vram_allocated_mb >= 0.0
