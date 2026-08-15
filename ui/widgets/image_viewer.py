"""Interactive image workspace with zoom, pan, and overlays."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDragEnterEvent, QDropEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from core.contracts.analysis import SceneGraph
from core.contracts.pipeline import PipelineResult
from ui.components.button import SentivisButton
from ui.design import DARK_TOKENS
from ui.i18n.translator import tr
from ui.preferences.ui_preferences import load_comparison_mode


class _OverlayRect(QGraphicsRectItem):
    """Detection box with hover tooltip and click highlight."""

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        node_index: int,
        label: str,
        confidence: float,
        color: QColor,
        on_select: Callable[[int], None],
    ) -> None:
        super().__init__(x, y, w, h)
        self._node_index = node_index
        self._label = label
        self._confidence = confidence
        self._base_color = color
        self._on_select = on_select
        self._highlighted = False
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.GlobalColor.transparent))
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    @property
    def node_index(self) -> int:
        return self._node_index

    def hoverEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        QToolTip.showText(
            event.screenPos(),
            f"{self._label} ({self._confidence:.0%})",
        )
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self._on_select(self._node_index)
        super().mousePressEvent(event)

    def set_highlighted(self, highlighted: bool) -> None:
        self._highlighted = highlighted
        width = 3 if highlighted else 2
        color = QColor(DARK_TOKENS.secondary if highlighted else self._base_color)
        self.setPen(QPen(color, width))


class _OverlayLine(QGraphicsLineItem):
    """Relationship line with click highlight."""

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        relation_type: str,
        subject_index: int,
        object_index: int,
        on_select: Callable[[int, int], None],
    ) -> None:
        super().__init__(x1, y1, x2, y2)
        self._relation_type = relation_type
        self._subject_index = subject_index
        self._object_index = object_index
        self._on_select = on_select
        self.setPen(QPen(QColor(DARK_TOKENS.accent), 1, Qt.PenStyle.DashLine))
        self.setAcceptHoverEvents(True)
        self.setZValue(9)

    @property
    def subject_index(self) -> int:
        return self._subject_index

    @property
    def object_index(self) -> int:
        return self._object_index

    def hoverEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        QToolTip.showText(event.screenPos(), self._relation_type.replace("_", " "))
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self._on_select(self._subject_index, self._object_index)
        super().mousePressEvent(event)

    def set_highlighted(self, highlighted: bool) -> None:
        width = 3 if highlighted else 1
        color = QColor(DARK_TOKENS.secondary if highlighted else DARK_TOKENS.accent)
        self.setPen(QPen(color, width, Qt.PenStyle.DashLine))


class ImageViewerWidget(QWidget):
    """Center workspace for image preview, navigation, and detection overlays."""

    image_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setProperty("class", "Card")
        self._source_pixmap = QPixmap()
        self._image_path: Path | None = None
        self._scale_factor = 1.0
        self._overlay_items: list[QGraphicsRectItem | QGraphicsLineItem | QGraphicsSimpleTextItem] = []
        self._box_items: list[_OverlayRect] = []
        self._line_items: list[_OverlayLine] = []
        self._node_centers: dict[int, tuple[float, float]] = {}
        self._last_result: PipelineResult | None = None
        self._toolbar_widget = QWidget()
        self._overlay_colors = (DARK_TOKENS.secondary, DARK_TOKENS.accent)
        self._selected_node: int | None = None
        self._selected_relation: tuple[int, int] | None = None
        self._enhanced_preview_path: Path | None = None
        self._comparison_enabled = load_comparison_mode()

        root = QVBoxLayout(self)
        toolbar = QHBoxLayout(self._toolbar_widget)
        self._meta_label = QLabel("")
        self._meta_label.setProperty("class", "StatusLabel")
        self._meta_label.setWordWrap(True)
        self._fit_button = SentivisButton(tr("button.fit"), variant="secondary")
        self._original_button = SentivisButton(tr("button.original"), variant="secondary")
        self._zoom_in_button = SentivisButton(tr("button.zoom_in"), variant="secondary")
        self._zoom_out_button = SentivisButton(tr("button.zoom_out"), variant="secondary")
        self._fit_button.setToolTip(tr("tooltip.fit"))
        self._zoom_in_button.setToolTip(tr("tooltip.zoom_in"))
        self._zoom_out_button.setToolTip(tr("tooltip.zoom_out"))
        self._detection_toggle = QCheckBox(tr("overlay.detections"))
        self._detection_toggle.setChecked(False)
        self._relationship_toggle = QCheckBox(tr("overlay.relationships"))
        self._relationship_toggle.setChecked(False)
        self._activities_toggle = QCheckBox(tr("overlay.activities"))
        self._activities_toggle.setChecked(False)
        self._labels_toggle = QCheckBox(tr("overlay.labels"))
        self._labels_toggle.setChecked(True)
        self._heatmap_toggle = QCheckBox(tr("overlay.heatmap"))
        self._heatmap_toggle.setChecked(False)
        self._attention_toggle = QCheckBox(tr("overlay.attention"))
        self._attention_toggle.setChecked(False)
        self._comparison_toggle = QCheckBox(tr("overlay.comparison"))
        self._comparison_toggle.setChecked(self._comparison_enabled)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setEnabled(False)
        self._opacity_label = QLabel(tr("overlay.opacity"))
        for widget in (
            self._fit_button,
            self._original_button,
            self._zoom_in_button,
            self._zoom_out_button,
            self._detection_toggle,
            self._relationship_toggle,
            self._activities_toggle,
            self._labels_toggle,
            self._heatmap_toggle,
            self._attention_toggle,
            self._comparison_toggle,
            self._opacity_label,
            self._opacity_slider,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        root.addWidget(self._toolbar_widget)
        root.addWidget(self._meta_label)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._pixmap_item = QGraphicsPixmapItem()
        self._enhanced_pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._scene.addItem(self._enhanced_pixmap_item)
        self._enhanced_pixmap_item.setZValue(2)
        self._enhanced_pixmap_item.setOpacity(0.5)
        self._enhanced_pixmap_item.setVisible(False)

        self._placeholder = QLabel(tr("image.placeholder"))
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setProperty("class", "Placeholder")
        root.addWidget(self._placeholder, stretch=0)
        root.addWidget(self._view, stretch=1)

        self._fit_button.clicked.connect(self.fit_to_window)
        self._original_button.clicked.connect(self.show_original_size)
        self._zoom_in_button.clicked.connect(lambda: self._apply_zoom(1.12))
        self._zoom_out_button.clicked.connect(lambda: self._apply_zoom(1 / 1.12))
        for toggle in (
            self._detection_toggle,
            self._relationship_toggle,
            self._activities_toggle,
            self._labels_toggle,
            self._heatmap_toggle,
            self._attention_toggle,
            self._comparison_toggle,
        ):
            toggle.toggled.connect(self._on_overlay_toggle)

        self._opacity_slider.valueChanged.connect(self._update_comparison_opacity)

    def retranslate_ui(self) -> None:
        self._fit_button.setText(tr("button.fit"))
        self._original_button.setText(tr("button.original"))
        self._zoom_in_button.setText(tr("button.zoom_in"))
        self._zoom_out_button.setText(tr("button.zoom_out"))
        self._detection_toggle.setText(tr("overlay.detections"))
        self._relationship_toggle.setText(tr("overlay.relationships"))
        self._activities_toggle.setText(tr("overlay.activities"))
        self._labels_toggle.setText(tr("overlay.labels"))
        self._heatmap_toggle.setText(tr("overlay.heatmap"))
        self._attention_toggle.setText(tr("overlay.attention"))
        self._comparison_toggle.setText(tr("overlay.comparison"))
        self._opacity_label.setText(tr("overlay.opacity"))
        if self._source_pixmap.isNull():
            self._placeholder.setText(tr("image.placeholder"))

    def set_presentation_mode(self, enabled: bool) -> None:
        self._toolbar_widget.setVisible(not enabled)
        self._meta_label.setVisible(not enabled)
        if enabled:
            self._detection_toggle.setChecked(False)
            self._relationship_toggle.setChecked(False)
            self._rebuild_overlays()

    def zoom_in(self) -> None:
        self._apply_zoom(1.12)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.12)

    def set_loading(self, active: bool) -> None:
        if active:
            self._placeholder.setText(tr("image.loading"))
            self._placeholder.show()
        elif self._source_pixmap.isNull():
            self._placeholder.setText(tr("image.placeholder"))
            self._placeholder.show()
        else:
            self._placeholder.hide()

    def show_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self._image_path = path
        self._source_pixmap = pixmap
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._placeholder.hide()
        self._view.show()
        self.fit_to_window()
        self.clear_overlays()
        self._update_meta(status="Image ready")

    def clear_image(self) -> None:
        self._image_path = None
        self._source_pixmap = QPixmap()
        self._pixmap_item.setPixmap(QPixmap())
        self._enhanced_pixmap_item.setPixmap(QPixmap())
        self._enhanced_pixmap_item.setVisible(False)
        self._enhanced_preview_path = None
        self._last_result = None
        self.clear_overlays()
        self._meta_label.setText("")
        self._placeholder.setText(tr("image.placeholder"))
        self._placeholder.show()

    def _update_meta(self, *, status: str, quality: str | None = None) -> None:
        if self._source_pixmap.isNull():
            self._meta_label.setText("")
            return
        width = self._source_pixmap.width()
        height = self._source_pixmap.height()
        quality_text = quality or "—"
        self._meta_label.setText(
            f"Resolution: {width}×{height}   ·   Quality Score: {quality_text}   ·   Processing: {status}"
        )


    def set_comparison_mode(self, enabled: bool) -> None:
        self._comparison_enabled = enabled
        self._comparison_toggle.setChecked(enabled)
        self._apply_comparison_view()

    def show_result_overlays(self, result: PipelineResult | None) -> None:
        self._last_result = result
        self._enhanced_preview_path = result.enhanced_preview_path if result else None
        has_enhanced = self._enhanced_preview_path is not None and self._enhanced_preview_path.is_file()
        self._comparison_toggle.setEnabled(has_enhanced)
        self._opacity_slider.setEnabled(has_enhanced and self._comparison_toggle.isChecked())
        self._apply_comparison_view()
        self._rebuild_overlays()
        quality = None
        if result is not None and result.image_quality is not None:
            score = result.image_quality.after_quality or result.image_quality.metrics.estimated_quality
            quality = f"{float(score):.0%}" if float(score) <= 1.0 else f"{float(score):.0f}"
        elif result is not None:
            quality = f"{result.quality_report.overall_quality:.0%}"
        self._update_meta(status="Complete" if result else "Image ready", quality=quality)

    def _on_overlay_toggle(self, _checked: bool) -> None:
        if self.sender() is self._comparison_toggle:
            self._opacity_slider.setEnabled(
                self._comparison_toggle.isChecked() and self._enhanced_preview_path is not None
            )
            self._apply_comparison_view()
        self._rebuild_overlays()

    def _update_comparison_opacity(self, value: int) -> None:
        self._enhanced_pixmap_item.setOpacity(max(0.0, min(1.0, value / 100.0)))

    def _apply_comparison_view(self) -> None:
        if (
            self._comparison_toggle.isChecked()
            and self._enhanced_preview_path is not None
            and self._enhanced_preview_path.is_file()
        ):
            enhanced = QPixmap(str(self._enhanced_preview_path))
            if not enhanced.isNull():
                self._enhanced_pixmap_item.setPixmap(enhanced)
                self._enhanced_pixmap_item.setVisible(True)
                self._update_comparison_opacity(self._opacity_slider.value())
                return
        self._enhanced_pixmap_item.setVisible(False)

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._box_items.clear()
        self._line_items.clear()
        self._node_centers.clear()
        self._selected_node = None
        self._selected_relation = None

    def fit_to_window(self) -> None:
        if self._source_pixmap.isNull():
            return
        self._view.resetTransform()
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._scale_factor = self._view.transform().m11()

    def show_original_size(self) -> None:
        if self._source_pixmap.isNull():
            return
        self._view.resetTransform()
        self._scale_factor = 1.0

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self.image_dropped.emit(str(path))
                break

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._source_pixmap.isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1 / 1.12
        self._apply_zoom(factor)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().resizeEvent(event)
        if self._scale_factor <= 1.01 and not self._source_pixmap.isNull():
            self.fit_to_window()

    def _apply_zoom(self, factor: float) -> None:
        self._scale_factor = max(0.05, min(12.0, self._scale_factor * factor))
        self._view.scale(factor, factor)

    def _select_node(self, node_index: int) -> None:
        self._selected_node = node_index
        self._selected_relation = None
        self._apply_highlights()

    def _select_relation(self, subject_index: int, object_index: int) -> None:
        self._selected_relation = (subject_index, object_index)
        self._selected_node = None
        self._apply_highlights()

    def _apply_highlights(self) -> None:
        for box in self._box_items:
            highlighted = self._selected_node == box.node_index
            if self._selected_relation is not None:
                subj, obj = self._selected_relation
                highlighted = box.node_index in {subj, obj}
            box.set_highlighted(highlighted)
        for line in self._line_items:
            highlighted = False
            if self._selected_relation is not None:
                subj, obj = self._selected_relation
                highlighted = line.subject_index == subj and line.object_index == obj
            elif self._selected_node is not None:
                highlighted = (
                    line.subject_index == self._selected_node
                    or line.object_index == self._selected_node
                )
            line.set_highlighted(highlighted)

    def _rebuild_overlays(self) -> None:
        self.clear_overlays()
        result = self._last_result
        if result is None or self._source_pixmap.isNull():
            return
        width = float(self._source_pixmap.width())
        height = float(self._source_pixmap.height())
        graph = result.scene_context.graph
        if self._detection_toggle.isChecked() or self._labels_toggle.isChecked():
            self._draw_detection_overlays(graph, width, height)
        if self._relationship_toggle.isChecked():
            self._draw_relationship_overlays(graph)
        if self._activities_toggle.isChecked():
            self._draw_activity_markers(result, width, height)
        if self._heatmap_toggle.isChecked() or self._attention_toggle.isChecked():
            self._draw_attention_overlay(graph, width, height, heatmap=self._heatmap_toggle.isChecked())

    def _draw_detection_overlays(self, graph: SceneGraph, width: float, height: float) -> None:
        for node in graph.nodes:
            x_min, y_min, box_w, box_h = _approximate_box(
                node.position_zone,
                node.bounding_box_area_ratio,
                width,
                height,
            )
            score = min(1.0, max(0.2, node.bounding_box_area_ratio * 4.0))
            rect = _OverlayRect(
                x_min,
                y_min,
                box_w,
                box_h,
                node_index=node.index,
                label=node.label,
                confidence=score,
                color=QColor(self._overlay_colors[0]),
                on_select=self._select_node,
            )
            if not self._detection_toggle.isChecked():
                rect.setPen(QPen(Qt.PenStyle.NoPen))
            self._scene.addItem(rect)
            self._overlay_items.append(rect)
            self._box_items.append(rect)
            self._node_centers[node.index] = (x_min + box_w / 2, y_min + box_h / 2)
            if self._labels_toggle.isChecked():
                label_item = QGraphicsSimpleTextItem(
                    f"{node.label} {min(1.0, node.bounding_box_area_ratio * 4):.0%}"
                )
                label_item.setPos(x_min, max(0.0, y_min - 16))
                label_item.setBrush(QBrush(QColor(DARK_TOKENS.text_primary)))
                label_item.setZValue(11)
                self._scene.addItem(label_item)
                self._overlay_items.append(label_item)

    def _draw_relationship_overlays(self, graph: SceneGraph) -> None:
        for relation in graph.relations:
            start = self._node_centers.get(relation.subject_index)
            end = self._node_centers.get(relation.object_index)
            if start is None or end is None:
                continue
            line = _OverlayLine(
                start[0],
                start[1],
                end[0],
                end[1],
                relation_type=relation.relation_type,
                subject_index=relation.subject_index,
                object_index=relation.object_index,
                on_select=self._select_relation,
            )
            self._scene.addItem(line)
            self._overlay_items.append(line)
            self._line_items.append(line)

    def _draw_activity_markers(self, result: PipelineResult, width: float, height: float) -> None:
        activities = result.scene_context.activities.activities
        if not activities:
            return
        for index, evidence in enumerate(activities[:5]):
            if not evidence.supporting_node_indices:
                continue
            node_index = evidence.supporting_node_indices[0]
            center = self._node_centers.get(node_index)
            if center is None:
                continue
            marker = QGraphicsSimpleTextItem(f"▶ {evidence.activity}")
            marker.setPos(center[0], center[1] + 8 + index * 14)
            marker.setBrush(QBrush(QColor(DARK_TOKENS.success)))
            marker.setZValue(12)
            self._scene.addItem(marker)
            self._overlay_items.append(marker)

    def _draw_attention_overlay(
        self,
        graph: SceneGraph,
        width: float,
        height: float,
        *,
        heatmap: bool,
    ) -> None:
        if not graph.nodes:
            return
        for node in graph.nodes:
            x_min, y_min, box_w, box_h = _approximate_box(
                node.position_zone,
                node.bounding_box_area_ratio,
                width,
                height,
            )
            alpha = int(40 + min(1.0, node.bounding_box_area_ratio * 4) * (120 if heatmap else 60))
            color = QColor(DARK_TOKENS.accent if heatmap else DARK_TOKENS.primary)
            color.setAlpha(min(180, alpha))
            rect = QGraphicsRectItem(x_min, y_min, box_w, box_h)
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.setBrush(QBrush(color))
            rect.setZValue(5)
            self._scene.addItem(rect)
            self._overlay_items.append(rect)


def _approximate_box(
    zone: str,
    area_ratio: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """Map scene node zone metadata to an approximate overlay rectangle."""
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
