"""Final caption lock and execution diagnostics."""

from __future__ import annotations

from core.contracts.language import RefinedCaption
from core.contracts.metrics import PipelineMetrics
from core.constants.pipeline_stages import PipelineStage


def test_refined_caption_stores_canonical_english() -> None:
    caption = RefinedCaption(
        text="Eine Person steht im Buero.",
        sources=("natural_caption",),
        narrative_full="Eine Person steht im Buero.",
        canonical_caption_en="A person stands in an office.",
    )
    assert caption.canonical_caption_en.startswith("A person")
    assert caption.text != caption.canonical_caption_en


def test_pipeline_metrics_include_execution_counts() -> None:
    metrics = PipelineMetrics(
        total_duration_ms=10.0,
        peak_ram_mb=1.0,
        peak_vram_mb=0.0,
        stage_metrics=(),
        model_timings=(),
        objects_detected=1,
        relationships_inferred=0,
        activities_inferred=0,
        scene_graph_nodes=1,
        scene_graph_edges=0,
        caption_quality_score=0.9,
        recovery_events=0,
        fallback_events=0,
        competition_mode=False,
        qa_passed=True,
        vlm_executions=1,
        caption_generation_count=1,
        qa_count=1,
    )
    assert metrics.vlm_executions == 1
    assert metrics.caption_generation_count == 1
    assert metrics.qa_count == 1
    assert PipelineStage.QUALITY_EVALUATION.value


def test_supported_ui_languages_match_specification() -> None:
    from streamlit_app.catalog import SUPPORTED_LANGUAGES
    from language.localization.caption_translator import SUPPORTED_CAPTION_LANGUAGES

    assert SUPPORTED_LANGUAGES == ("en", "fa", "de", "es", "zh")
    assert SUPPORTED_CAPTION_LANGUAGES == ("en", "fa", "de", "es", "zh")
