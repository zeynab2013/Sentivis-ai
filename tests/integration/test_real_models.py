"""Real production model integration tests (requires installed models)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.container import ApplicationContext, DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from model_management.service import ModelManagementService


def _scene(path: Path, *, kind: str) -> Path:
    image = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    if kind == "people":
        draw.ellipse((280, 120, 360, 220), fill=(220, 180, 140))
        draw.rectangle((260, 220, 380, 420), fill=(40, 80, 200))
    elif kind == "vehicles":
        draw.rectangle((120, 260, 520, 380), fill=(200, 40, 40))
        draw.ellipse((150, 360, 210, 420), fill=(20, 20, 20))
        draw.ellipse((430, 360, 490, 420), fill=(20, 20, 20))
    elif kind == "landscape":
        draw.rectangle((0, 300, 640, 480), fill=(40, 120, 40))
        draw.polygon([(0, 300), (320, 80), (640, 300)], fill=(100, 160, 220))
    else:
        draw.rectangle((80, 80, 560, 400), fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


@pytest.fixture(scope="module")
def real_context() -> ApplicationContext:
    try:
        context = DependencyContainer().build(
            load_app_config(),
            load_model_config(),
            load_theme_config(),
            load_analysis_config(),
        )
        service = ModelManagementService.create(
            context.model_registry,
            context.main_controller.app_config.paths.models_dir,
        )
        if not service.all_mandatory_ready():
            pytest.skip("Production models not installed — run: sentivis-models download")
        return context
    except Exception as exc:
        pytest.skip(f"Real model context unavailable: {exc}")


@pytest.mark.real_models
@pytest.mark.parametrize(
    "scene",
    ["people", "vehicles", "indoor", "outdoor", "animals", "crowded", "landscape", "food", "low_light", "multi_object"],
)
def test_real_pipeline_scene(tmp_path: Path, real_context: ApplicationContext, scene: str) -> None:
    image = _scene(tmp_path / f"{scene}.png", kind=scene)
    orchestrator = real_context.main_controller.pipeline._orchestrator  # noqa: SLF001
    result = orchestrator.analyze(PipelineRequest(image, AnalysisOptions(enable_gemma=True)))
    assert result.caption.text
    assert result.metrics.total_duration_ms >= 0
    real_context.model_manager.release_all()
    real_context.memory_manager.clear_gpu_cache()
