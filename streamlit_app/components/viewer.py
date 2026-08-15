"""Image viewer with overlays and comparison for Streamlit."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from core.contracts.pipeline import PipelineResult

_OVERLAY_PURPLE = (109, 40, 217)
_OVERLAY_GOLD = (245, 197, 66)
_OVERLAY_BLUE = (56, 189, 248)


def load_display_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _approximate_box(
    zone: str,
    area_ratio: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    horizontal, vertical = _split_zone(zone)
    cell_w = image_width / 3.0
    cell_h = image_height / 3.0
    col = {"left": 0, "center": 1, "right": 2}.get(horizontal, 1)
    row = {"top": 0, "middle": 1, "bottom": 2}.get(vertical, 1)
    side = max(24.0, (area_ratio * image_width * image_height) ** 0.5)
    x_min = col * cell_w + (cell_w - side) / 2.0
    y_min = row * cell_h + (cell_h - side) / 2.0
    return x_min, y_min, side, side


def _split_zone(zone: str) -> tuple[str, str]:
    parts = zone.split("-", maxsplit=1)
    if len(parts) == 2:
        return parts[1], parts[0]
    return "center", "middle"


def render_overlays(
    base: Image.Image,
    result: PipelineResult | None,
    *,
    show_boxes: bool = True,
    show_relationships: bool = False,
    show_labels: bool = True,
    highlight_index: int | None = None,
    overlay_opacity: float = 0.5,
) -> Image.Image:
    if result is None:
        return base
    canvas = base.copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = canvas.size
    graph = result.scene_context.graph
    centers: dict[int, tuple[float, float]] = {}
    alpha = max(0.0, min(1.0, overlay_opacity))

    for node in graph.nodes:
        x_min, y_min, box_w, box_h = _approximate_box(
            node.position_zone,
            node.bounding_box_area_ratio,
            float(width),
            float(height),
        )
        centers[node.index] = (x_min + box_w / 2, y_min + box_h / 2)
        if show_boxes:
            color = _OVERLAY_GOLD if node.index == highlight_index else _OVERLAY_PURPLE
            fill_alpha = int(40 * alpha)
            draw.rectangle(
                (x_min, y_min, x_min + box_w, y_min + box_h),
                fill=(*color, fill_alpha),
                outline=(*color, int(220 * alpha)),
                width=3 if node.index == highlight_index else 2,
            )
        if show_labels:
            label = f"{node.label}"
            draw.rectangle(
                (x_min, max(0, y_min - 18), x_min + len(label) * 7 + 8, y_min),
                fill=(15, 17, 26, int(200 * alpha)),
            )
            draw.text((x_min + 4, max(2, y_min - 16)), label, fill=(236, 236, 241, 255))

    if show_relationships:
        for relation in graph.relations:
            start = centers.get(relation.subject_index)
            end = centers.get(relation.object_index)
            if start is None or end is None:
                continue
            draw.line([start, end], fill=(*_OVERLAY_BLUE, int(180 * alpha)), width=2)

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def render_minimap(base: Image.Image, max_size: int = 120) -> Image.Image:
    thumb = base.copy()
    thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    bordered = Image.new("RGB", (thumb.width + 4, thumb.height + 4), _OVERLAY_PURPLE)
    bordered.paste(thumb, (2, 2))
    return bordered
