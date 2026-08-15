"""Integration tests for language-stage failure recovery."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from certification.pipeline_stubs import StubDetector
from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import SceneContext
from core.contracts.image import PreprocessedImage
from core.contracts.language import Prompt, RawCaption, VisualObservations
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from core.exceptions.language import InferenceError
from tests.support.pipeline_harness import build_test_orchestrator


class FailingVisionLanguage:
    def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
        raise InferenceError(
            "Visual understanding failed during analysis.",
            "stub blip failure",
            stage=PipelineStage.BLIP_UNDERSTANDING,
            recoverable=True,
        )


class FailingReasoning:
    def reason(self, prompt: Prompt, context: SceneContext) -> RawCaption:
        raise InferenceError(
            "Reasoning failed during caption generation.",
            "stub gemma failure",
            stage=PipelineStage.GEMMA_REASONING,
            recoverable=True,
        )


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


def test_pipeline_continues_when_blip_fails(sample_image: Path) -> None:
    result = build_test_orchestrator(
        StubDetector(),
        FailingVisionLanguage(),
        FailingReasoning(),
    ).analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    assert result.caption.text
    assert result.warnings
    assert result.quality_report.overall_quality >= 0.0
    assert result.metrics.fallback_events >= 1


def test_pipeline_continues_when_gemma_fails(sample_image: Path) -> None:
    class StubVisionLanguage:
        def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
            raw = RawCaption(text="A person in a scene.", source="blip", confidence=0.9)
            return VisualObservations(
                observations=(raw.text,),
                object_attributes=(),
                candidate_descriptions=(raw.text,),
                confidence=0.9,
                raw_caption=raw,
            )

    orchestrator = build_test_orchestrator(
        StubDetector(),
        StubVisionLanguage(),
        FailingReasoning(),
    )
    # Force Gemma path so recovery is exercised even when Ollama semantic is available.
    semantic = orchestrator._semantic_reasoning
    semantic._semantic_cfg = replace(semantic._semantic_cfg, enabled=False, prefer_over_gemma=False)
    result = orchestrator.analyze(PipelineRequest(sample_image, AnalysisOptions(enable_gemma=True)))
    assert "person" in result.caption.text.lower()
    assert result.warnings
    assert result.metrics.fallback_events >= 1
