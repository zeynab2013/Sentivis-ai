"""Export writers and manager."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from core.contracts.pipeline import PipelineResult
from core.exceptions.service import ExportError
from core.logging import get_logger
from services.export.report_builder import report_sections
from ui.branding.logo_provider import branding_logo_path

logger = get_logger(__name__)

# Optional Vision Assistant transcript attached for the current export call.
_EXPORT_ASSISTANT_TRANSCRIPT = ""


def set_export_assistant_transcript(text: str) -> None:
    global _EXPORT_ASSISTANT_TRANSCRIPT
    _EXPORT_ASSISTANT_TRANSCRIPT = (text or "").strip()


def _sections_for(result: PipelineResult) -> dict[str, str]:
    return report_sections(result, assistant_transcript=_EXPORT_ASSISTANT_TRANSCRIPT)


class ExportWriter(ABC):
    """Base class for export format writers."""

    @abstractmethod
    def write(self, result: PipelineResult, path: Path) -> None:
        ...


class JsonExportWriter(ExportWriter):
    """Export pipeline result as JSON."""

    def write(self, result: PipelineResult, path: Path) -> None:
        sections = _sections_for(result)
        payload = {
            "image": str(result.request.image_path),
            "narrative_caption": sections["narrative_caption"],
            "narrative_full": sections["narrative_full"],
            "narrative_short": sections["narrative_short"],
            "executive_summary": sections.get("executive_summary", ""),
            "caption": sections["caption"],
            "sources": list(result.caption.sources),
            "scene_summary": sections["scene_summary"],
            "objects": sections["objects"],
            "object_details": sections["object_details"],
            "color_palette": sections["color_palette"],
            "scene_graph": sections["scene_graph"],
            "relationships": sections["relationships"],
            "activities": sections["activities"],
            "context": sections["context"],
            "image_quality": sections["image_quality"],
            "caption_confidence": sections["caption_confidence"],
            "quality_report": sections["quality"],
            "pipeline_metrics": sections["metrics"],
            "object_count": result.scene_context.object_count,
            "dominant_objects": list(result.scene_context.dominant_objects),
            "warnings": list(result.warnings),
            "qa_passed": result.qa_passed,
        }
        if sections.get("vision_assistant"):
            payload["vision_assistant"] = sections["vision_assistant"]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TxtExportWriter(ExportWriter):
    """Export full analysis report as plain text."""

    def write(self, result: PipelineResult, path: Path) -> None:
        sections = _sections_for(result)
        lines = [
            "Sentivis AI Analysis Report",
            "===========================",
            f"Image: {result.request.image_path.name}",
            "",
            "Narrative Caption",
            "-----------------",
            sections["narrative_caption"],
            "",
            "Caption",
            "-------",
            sections["caption"],
            "",
            "Scene Summary",
            "-------------",
            sections["scene_summary"],
            "",
            "Detected Objects",
            "----------------",
            sections["objects"],
            "",
            "Object Details",
            "--------------",
            sections["object_details"],
            "",
            "Color Palette",
            "-------------",
            sections["color_palette"],
            "",
            "Scene Graph",
            "-----------",
            sections["scene_graph"],
            "",
            "Relationships",
            "-------------",
            sections["relationships"],
            "",
            "Activities",
            "----------",
            sections["activities"],
            "",
            "Context",
            "-------",
            sections["context"],
            "",
            "Image Quality",
            "-------------",
            sections["image_quality"],
            "",
            "Caption Confidence",
            "------------------",
            sections["caption_confidence"],
            "",
            "Caption Quality",
            "---------------",
            sections["quality"],
            "",
            "Pipeline Metrics",
            "----------------",
            sections["metrics"],
        ]
        if sections.get("vision_assistant"):
            lines.extend(
                [
                    "",
                    "Vision Assistant",
                    "----------------",
                    sections["vision_assistant"],
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")


class MarkdownExportWriter(ExportWriter):
    """Export analysis report as Markdown."""

    def write(self, result: PipelineResult, path: Path) -> None:
        sections = _sections_for(result)
        lines = [
            "# Sentivis AI Analysis Report",
            "",
            f"**Image:** `{result.request.image_path.name}`",
            "",
            "## Narrative Caption",
            "",
            sections["narrative_full"],
            "",
            "**Executive Summary:**",
            "",
            sections.get("executive_summary", sections["narrative_short"]),
            "",
            "**Short Caption:**",
            "",
            sections["narrative_short"],
            "",
            "## Caption",
            "",
            sections["caption"],
            "",
            "## Scene Summary",
            "",
            "```",
            sections["scene_summary"],
            "```",
            "",
            "## Detected Objects",
            "",
            sections["objects"],
            "",
            "## Object Details",
            "",
            "```",
            sections["object_details"],
            "```",
            "",
            "## Color Palette",
            "",
            sections["color_palette"],
            "",
            "## Scene Graph",
            "",
            "```",
            sections["scene_graph"],
            "```",
            "",
            "## Relationships",
            "",
            sections["relationships"],
            "",
            "## Activities",
            "",
            sections["activities"],
            "",
            "## Context",
            "",
            "```",
            sections["context"],
            "```",
            "",
            "## Image Quality",
            "",
            "```",
            sections["image_quality"],
            "```",
            "",
            "## Caption Confidence",
            "",
            "```",
            sections["caption_confidence"],
            "```",
            "",
            "## Caption Quality",
            "",
            "```",
            sections["quality"],
            "```",
            "",
            "## Pipeline Metrics",
            "",
            "```",
            sections["metrics"],
            "```",
        ]
        if sections.get("vision_assistant"):
            lines.extend(
                [
                    "",
                    "## Vision Assistant",
                    "",
                    "```",
                    sections["vision_assistant"],
                    "```",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")


class PdfExportWriter(ExportWriter):
    """Export analysis summary report as PDF."""

    def write(self, result: PipelineResult, path: Path) -> None:
        sections = _sections_for(result)
        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        page_number = 1
        y = height - 72

        def new_page() -> float:
            nonlocal page_number, y
            _draw_footer(pdf, width, page_number)
            pdf.showPage()
            page_number += 1
            return float(height - 72)

        logo_path = branding_logo_path()
        if logo_path.is_file():
            try:
                pdf.drawImage(
                    ImageReader(str(logo_path)),
                    72,
                    y - 8,
                    width=48,
                    height=48,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                logger.debug("PDF logo render skipped", exc_info=True)

        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(132, y, "Sentivis AI")
        y -= 22
        pdf.setFont("Helvetica", 10)
        pdf.drawString(132, y, "Visual Understanding Report")
        y -= 18
        pdf.drawString(
            72,
            y,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  Image: {result.request.image_path.name}",
        )
        y -= 28

        from language.refinement.caption_refiner import ui_text

        for heading, body in (
            (ui_text("section.executive", "Executive Summary"), sections.get("executive_summary", "")),
            (ui_text("export.section.narrative", "Narrative Caption"), sections["narrative_full"]),
            (ui_text("export.section.short", "Short Caption"), sections["narrative_short"]),
            (ui_text("section.objects", "Detected Objects"), sections["objects"]),
            (ui_text("streamlit.section.object_details", "Object Details"), sections["object_details"]),
            (ui_text("streamlit.section.color_palette", "Color Palette"), sections["color_palette"]),
            (ui_text("streamlit.section.scene_graph", "Scene Graph"), sections["scene_graph"]),
            (ui_text("section.activities", "Activities"), sections["activities"]),
            (ui_text("section.relationships", "Relationships"), sections["relationships"]),
            (ui_text("section.environment", "Environment"), sections["context"]),
            (ui_text("section.image_quality", "Image Quality"), sections["image_quality"]),
            (ui_text("export.section.confidence", "Caption Confidence"), sections["caption_confidence"]),
            (ui_text("section.quality", "Caption Quality"), sections["quality"]),
            (ui_text("section.metrics", "Pipeline Metrics"), sections["metrics"]),
        ):
            if not body.strip():
                continue
            y = _draw_section(pdf, heading, body, y, height, new_page)

        _draw_footer(pdf, width, page_number)
        pdf.save()


class HtmlExportWriter(ExportWriter):
    """Export analysis report as HTML."""

    def write(self, result: PipelineResult, path: Path) -> None:
        from language.refinement.caption_refiner import active_ui_language, ui_text

        sections = _sections_for(result)
        logo_path = branding_logo_path()
        logo_uri = logo_path.as_uri() if logo_path.is_file() else ""
        lang = active_ui_language()
        h_exec = _html_escape(ui_text("section.executive", "Executive Summary"))
        h_narr = _html_escape(ui_text("export.section.narrative", "Narrative Caption"))
        h_short = _html_escape(ui_text("export.section.short", "Short Caption"))
        h_obj = _html_escape(ui_text("section.objects", "Detected Objects"))
        h_det = _html_escape(ui_text("streamlit.section.object_details", "Object Details"))
        h_color = _html_escape(ui_text("streamlit.section.color_palette", "Color Palette"))
        h_graph = _html_escape(ui_text("streamlit.section.scene_graph", "Scene Graph"))
        h_act = _html_escape(ui_text("section.activities", "Activities"))
        h_rel = _html_escape(ui_text("section.relationships", "Relationships"))
        h_env = _html_escape(ui_text("section.environment", "Environment"))
        h_iq = _html_escape(ui_text("section.image_quality", "Image Quality"))
        h_conf = _html_escape(ui_text("export.section.confidence", "Caption Confidence"))
        h_qual = _html_escape(ui_text("section.quality", "Caption Quality"))
        h_met = _html_escape(ui_text("section.metrics", "Pipeline Metrics"))
        html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8"/>
  <title>Sentivis AI Report — {result.request.image_path.name}</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; background:#130A24; color:#F8F7FC; margin:2rem; line-height:1.55; }}
    .card {{ background:#22163E; border:1px solid rgba(255,63,164,0.35); border-left:3px solid #FF3FA4; border-radius:12px; padding:1.35rem 1.4rem; margin-bottom:1.1rem; }}
    h1 {{ color:#FF3FA4; letter-spacing:-0.02em; }} h2 {{ color:#FF63B8; font-size:1.1rem; margin-top:0; }}
    footer {{ color:#B9B4CC; font-size:0.85rem; margin-top:2rem; }}
  </style>
</head>
<body>
  <header>
    {"<img src='" + logo_uri + "' alt='Logo' height='48'/>" if logo_uri else ""}
    <h1>Sentivis AI</h1>
    <p>Image: {result.request.image_path.name} · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </header>
  <section class="card"><h2>{h_exec}</h2><p>{_html_escape(sections.get('executive_summary', ''))}</p></section>
  <section class="card"><h2>{h_narr}</h2><p>{_html_escape(sections['narrative_full'])}</p></section>
  <section class="card"><h2>{h_short}</h2><p>{_html_escape(sections['narrative_short'])}</p></section>
  <section class="card"><h2>{h_obj}</h2><pre>{_html_escape(sections['objects'])}</pre></section>
  <section class="card"><h2>{h_det}</h2><pre>{_html_escape(sections['object_details'])}</pre></section>
  <section class="card"><h2>{h_color}</h2><pre>{_html_escape(sections['color_palette'])}</pre></section>
  <section class="card"><h2>{h_graph}</h2><pre>{_html_escape(sections['scene_graph'])}</pre></section>
  <section class="card"><h2>{h_act}</h2><pre>{_html_escape(sections['activities'])}</pre></section>
  <section class="card"><h2>{h_rel}</h2><pre>{_html_escape(sections['relationships'])}</pre></section>
  <section class="card"><h2>{h_env}</h2><pre>{_html_escape(sections['context'])}</pre></section>
  <section class="card"><h2>{h_iq}</h2><pre>{_html_escape(sections['image_quality'])}</pre></section>
  <section class="card"><h2>{h_conf}</h2><pre>{_html_escape(sections['caption_confidence'])}</pre></section>
  <section class="card"><h2>{h_qual}</h2><pre>{_html_escape(sections['quality'])}</pre></section>
  <section class="card"><h2>{h_met}</h2><pre>{_html_escape(sections['metrics'])}</pre></section>
  <footer>Sentivis AI · Page 1</footer>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")


class ImageExportWriter(ExportWriter):
    """Copy source image to export destination."""

    def write(self, result: PipelineResult, path: Path) -> None:
        source = result.request.image_path
        path.write_bytes(source.read_bytes())


def _draw_section(
    pdf: canvas.Canvas,
    heading: str,
    body: str,
    y: float,
    height: float,
    new_page: Callable[[], float],
) -> float:
    if y < 96:
        y = new_page()
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(72, y, heading)
    y -= 18
    pdf.setFont("Helvetica", 10)
    for line in body.splitlines():
        for wrapped in _wrap_text(line, 90):
            if y < 72:
                y = new_page()
            pdf.drawString(84, y, wrapped)
            y -= 14
    y -= 10
    return y


def _draw_footer(pdf: canvas.Canvas, width: float, page_number: int) -> None:
    pdf.setFont("Helvetica", 8)
    pdf.drawString(72, 36, "Sentivis AI — Visual Understanding Report")
    pdf.drawRightString(width - 72, 36, f"Page {page_number}")


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


class ExportManager:
    """Routes export requests to format-specific writers."""

    def __init__(self) -> None:
        self._writers: dict[str, ExportWriter] = {
            "json": JsonExportWriter(),
            "txt": TxtExportWriter(),
            "md": MarkdownExportWriter(),
            "markdown": MarkdownExportWriter(),
            "pdf": PdfExportWriter(),
            "html": HtmlExportWriter(),
            "image": ImageExportWriter(),
        }

    def export(self, result: PipelineResult, export_format: str, path: Path) -> None:
        writer = self._writers.get(export_format.lower())
        if writer is None:
            raise ExportError(
                f"Export format '{export_format}' is not supported.",
                f"Unknown export format: {export_format}",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer.write(result, path)
            logger.info("Exported %s to %s", export_format, path)
        except OSError as exc:
            raise ExportError(
                "Export failed. Please choose a different location.",
                f"Export IO error: {exc}",
            ) from exc
