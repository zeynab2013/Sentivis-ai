"""Shared report section builders for export writers."""

from __future__ import annotations

from core.contracts.pipeline import PipelineResult
from language.refinement.caption_refiner import ui_text
from ui.formatters.result_formatters import (
    format_activities,
    format_caption_confidence,
    format_color_palette,
    format_detected_objects,
    format_environment,
    format_execution_metrics,
    format_image_quality,
    format_object_details,
    format_quality_report,
    format_relationships,
    format_scene_graph,
    format_scene_summary,
)


def report_sections(
    result: PipelineResult,
    *,
    assistant_transcript: str = "",
) -> dict[str, str]:
    """Build labeled report sections from a pipeline result (UI language)."""
    from language.semantic.narrative_generator import (
        executive_summary_from_paragraph,
        short_caption_from_paragraph,
    )

    # Display caption is the active-language translation of canonical English.
    # Export must never re-run vision analysis.
    narrative_full = result.caption.narrative_full.strip() or result.caption.text
    # Always derive short/executive from the full narrative so mid-sentence
    # truncation and identical-summary bugs cannot ship in reports.
    narrative_short = short_caption_from_paragraph(narrative_full)
    executive = executive_summary_from_paragraph(narrative_full)
    narrative_block = (
        f"{ui_text('label.full_caption', 'FULL CAPTION')}\n{narrative_full}\n\n"
        f"{ui_text('label.short_caption', 'Short Caption')}\n{narrative_short}\n\n"
        f"{ui_text('label.executive_summary', 'Executive Summary')}\n{executive}"
    )

    sections = {
        "narrative_caption": narrative_block,
        "narrative_full": narrative_full,
        "narrative_short": narrative_short,
        "executive_summary": executive,
        "caption": result.caption.text,
        "scene_summary": format_scene_summary(result),
        "objects": format_detected_objects(result),
        "object_details": format_object_details(result),
        "color_palette": format_color_palette(result),
        "scene_graph": format_scene_graph(result),
        "relationships": format_relationships(result),
        "activities": format_activities(result),
        "context": format_environment(result),
        "image_quality": format_image_quality(result),
        "caption_confidence": format_caption_confidence(result),
        "quality": format_quality_report(result),
        "metrics": format_execution_metrics(result),
    }
    transcript = (assistant_transcript or "").strip()
    if transcript:
        sections["vision_assistant"] = transcript
    return sections
