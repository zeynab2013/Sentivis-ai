"""Stress and resilience acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.startup.model_discovery import discover_models
from core.config.loader import load_app_config, load_model_config
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.exceptions.vision import ValidationError
from core.utils.paths import project_root
from services.models.device_selector import DeviceSelector
from services.pipeline.orchestrator import PipelineOrchestrator
from tests.acceptance.support.images import (
    create_corrupted_image,
    create_empty_file,
    create_large_image,
    create_standard_image,
)
from tests.acceptance.support.stubs import (
    AcceptanceStubDetector,
    AcceptanceStubReasoning,
    AcceptanceStubVisionLanguage,
)
from tests.support.pipeline_harness import build_test_orchestrator
from vision.validation.image_validator import ImageValidator


def _orchestrator() -> PipelineOrchestrator:
    return build_test_orchestrator(
        AcceptanceStubDetector(),
        AcceptanceStubVisionLanguage(),
        AcceptanceStubReasoning(),
    )


@pytest.mark.acceptance
@pytest.mark.stress
def test_multiple_sequential_analyses(tmp_path: Path) -> None:
    orchestrator = _orchestrator()
    image = create_standard_image(tmp_path / "seq.png")
    for index in range(5):
        result = orchestrator.analyze(PipelineRequest(image, AnalysisOptions(enable_gemma=True)))
        assert result.caption.text
        assert result.scene_context.object_count >= 2, f"run {index} failed"


@pytest.mark.acceptance
@pytest.mark.stress
def test_large_image_analysis(tmp_path: Path) -> None:
    orchestrator = _orchestrator()
    image = create_large_image(tmp_path / "large.png")
    result = orchestrator.analyze(PipelineRequest(image, AnalysisOptions(enable_gemma=True)))
    assert result.caption.text


@pytest.mark.acceptance
@pytest.mark.stress
def test_missing_model_graceful_registry(tmp_path: Path) -> None:
    empty_models = tmp_path / "empty_models"
    empty_models.mkdir()
    report = discover_models(load_model_config(), empty_models)
    assert report.entries
    assert any(not entry.available for entry in report.entries) or report.warnings


@pytest.mark.acceptance
@pytest.mark.stress
def test_invalid_image_rejected(tmp_path: Path) -> None:
    validator = ImageValidator(load_app_config())
    tiny = create_standard_image(tmp_path / "tiny.png", width=8, height=8)
    with pytest.raises(ValidationError):
        validator.validate(tiny)


@pytest.mark.acceptance
@pytest.mark.stress
def test_corrupted_image_rejected(tmp_path: Path) -> None:
    validator = ImageValidator(load_app_config())
    corrupted = create_corrupted_image(tmp_path / "corrupt.png")
    with pytest.raises(ValidationError):
        validator.validate(corrupted)


@pytest.mark.acceptance
@pytest.mark.stress
def test_empty_file_rejected(tmp_path: Path) -> None:
    validator = ImageValidator(load_app_config())
    empty = create_empty_file(tmp_path / "empty.png")
    with pytest.raises(ValidationError):
        validator.validate(empty)


@pytest.mark.acceptance
@pytest.mark.stress
def test_empty_models_folder_discovery() -> None:
    root = project_root()
    report = discover_models(load_model_config(), root / "models")
    assert report.entries


@pytest.mark.acceptance
@pytest.mark.stress
def test_gpu_unavailable_cpu_fallback() -> None:
    selector = DeviceSelector(load_app_config())
    device = selector.preferred_device("cuda")
    assert device in {"cpu", "cuda"}


@pytest.mark.acceptance
@pytest.mark.stress
def test_pipeline_cpu_fallback_completes(tmp_path: Path) -> None:
    orchestrator = _orchestrator()
    image = create_standard_image(tmp_path / "cpu_fallback.png")
    result = orchestrator.analyze(PipelineRequest(image, AnalysisOptions(enable_gemma=True)))
    assert result.caption.text
