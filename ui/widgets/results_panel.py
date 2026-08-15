"""Structured analysis results with collapsible sections."""

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from ui.components.button import SentivisButton
from ui.components.empty_state import EmptyStateWidget
from ui.components.progress import SentivisProgressBar
from ui.components.scroll_panel import SentivisScrollPanel
from ui.i18n.translator import tr
from ui.widgets.collapsible_section import CollapsibleSection


class ResultsPanelWidget(QWidget):
    """Right-panel results: progress, search, copy, and detail sections."""

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._toolbar_row = QWidget()
        toolbar = QHBoxLayout(self._toolbar_row)
        toolbar.setContentsMargins(0, 0, 0, 0)
        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("results.search"))
        self._search.setToolTip(tr("tooltip.search"))
        self._search.textChanged.connect(self._apply_filter)
        self._copy_all_button = SentivisButton(tr("button.copy_all"), variant="secondary")
        self._copy_all_button.setToolTip(tr("tooltip.copy_all"))
        self._copy_all_button.clicked.connect(self._copy_all)
        self._expand_all_button = SentivisButton(tr("button.expand_all"), variant="secondary")
        self._expand_all_button.clicked.connect(self._expand_all)
        self._collapse_all_button = SentivisButton(tr("button.collapse_all"), variant="secondary")
        self._collapse_all_button.clicked.connect(self._collapse_all)
        toolbar.addWidget(self._search, stretch=1)
        toolbar.addWidget(self._copy_all_button)
        toolbar.addWidget(self._expand_all_button)
        toolbar.addWidget(self._collapse_all_button)

        self._status_label = QLabel(tr("status.ready"))
        self._status_label.setProperty("class", "StatusLabel")
        self._stage_label = QLabel("")
        self._stage_label.setProperty("class", "StatusLabel")
        self._progress_bar = SentivisProgressBar()

        scroll = SentivisScrollPanel()
        container = QWidget()
        layout = QVBoxLayout(container)

        self._empty_state = EmptyStateWidget(
            tr("results.empty.title"),
            tr("results.empty.body"),
            action_hint=tr("results.empty.action"),
        )

        self._narrative = CollapsibleSection(tr("section.narrative"), expanded=True)
        self._caption = CollapsibleSection(tr("section.final_caption"), expanded=True)
        self._scene_summary = CollapsibleSection(tr("section.scene_summary"), expanded=True)
        self._objects = CollapsibleSection(tr("section.objects"), expanded=False)
        self._relationships = CollapsibleSection(tr("section.relationships"), expanded=False)
        self._activities = CollapsibleSection(tr("section.activities"), expanded=False)
        self._environment = CollapsibleSection(tr("section.environment"), expanded=False)
        self._image_quality = CollapsibleSection(tr("section.image_quality"), expanded=False)
        self._quality = CollapsibleSection(tr("section.quality"), expanded=False)
        self._metrics = CollapsibleSection(tr("section.metrics"), expanded=False)
        self._executive = CollapsibleSection(tr("section.executive"), expanded=True)
        self._short_caption = CollapsibleSection(tr("section.short_caption"), expanded=True)
        self._scene_description = CollapsibleSection(tr("section.scene_description"), expanded=True)
        self._attributes = CollapsibleSection(tr("section.attributes"), expanded=False)
        self._reasoning = CollapsibleSection(tr("section.reasoning"), expanded=False)
        self._confidence = CollapsibleSection(tr("section.confidence"), expanded=False)
        self._sections: tuple[CollapsibleSection, ...] = (
            self._executive,
            self._narrative,
            self._short_caption,
            self._scene_description,
            self._caption,
            self._scene_summary,
            self._objects,
            self._attributes,
            self._relationships,
            self._activities,
            self._environment,
            self._image_quality,
            self._reasoning,
            self._confidence,
            self._quality,
            self._metrics,
        )
        self._detail_sections: tuple[CollapsibleSection, ...] = (
            self._objects,
            self._attributes,
            self._relationships,
            self._activities,
            self._environment,
            self._image_quality,
            self._reasoning,
            self._confidence,
            self._quality,
            self._metrics,
        )

        for section in self._sections:
            section.copy_requested.connect(self._on_section_copied)

        layout.addWidget(self._empty_state)
        for section in self._sections:
            layout.addWidget(section)
        layout.addStretch()
        scroll.set_content(container)

        outer.addWidget(self._toolbar_row)
        outer.addWidget(self._status_label)
        outer.addWidget(self._stage_label)
        outer.addWidget(self._progress_bar)
        outer.addWidget(scroll, stretch=1)

        self._has_result = False
        self._is_analyzing = False
        self._presentation_mode = False

    @property
    def search_field(self) -> QLineEdit:
        return self._search

    def set_presentation_mode(self, enabled: bool) -> None:
        self._presentation_mode = enabled
        self.setProperty("presentation", enabled)
        self._toolbar_row.setVisible(not enabled)
        self._stage_label.setVisible(not enabled and bool(self._stage_label.text()))
        for section in self._detail_sections:
            section.setVisible(not enabled and section.isVisible())
        if enabled:
            self._narrative.expand()
            self._caption.expand()
            self._scene_summary.expand()
        self.style().unpolish(self)
        self.style().polish(self)

    def retranslate_ui(self) -> None:
        self._search.setPlaceholderText(tr("results.search"))
        self._search.setToolTip(tr("tooltip.search"))
        self._copy_all_button.setText(tr("button.copy_all"))
        self._copy_all_button.setToolTip(tr("tooltip.copy_all"))
        self._expand_all_button.setText(tr("button.expand_all"))
        self._collapse_all_button.setText(tr("button.collapse_all"))
        titles = (
            (self._narrative, "section.narrative"),
            (self._short_caption, "section.short_caption"),
            (self._executive, "section.executive"),
            (self._scene_description, "section.scene_description"),
            (self._caption, "section.final_caption"),
            (self._scene_summary, "section.scene_summary"),
            (self._objects, "section.objects"),
            (self._attributes, "section.attributes"),
            (self._relationships, "section.relationships"),
            (self._activities, "section.activities"),
            (self._environment, "section.environment"),
            (self._image_quality, "section.image_quality"),
            (self._reasoning, "section.reasoning"),
            (self._confidence, "section.confidence"),
            (self._quality, "section.quality"),
            (self._metrics, "section.metrics"),
        )
        for section, key in titles:
            section.set_title(tr(key))
        self._empty_state.set_content(
            tr("results.empty.title"),
            tr("results.empty.body"),
            action_hint=tr("results.empty.action"),
        )

    def bind(
        self,
        *,
        progress_percent: float,
        status_message: str,
        stage_label: str = "",
        has_result: bool,
        is_analyzing: bool,
        caption: str,
        narrative_text: str,
        short_caption_text: str,
        executive_text: str,
        scene_description_text: str,
        attributes_text: str,
        reasoning_text: str,
        confidence_text: str,
        scene_summary: str,
        objects_text: str,
        relationships_text: str,
        activities_text: str,
        environment_text: str,
        image_quality_text: str,
        quality_text: str,
        metrics_text: str,
    ) -> None:
        self._has_result = has_result
        self._is_analyzing = is_analyzing
        self._progress_bar.setValue(int(progress_percent))
        self._status_label.setText(status_message)
        self._stage_label.setText(stage_label)
        self._stage_label.setVisible(bool(stage_label) and not self._presentation_mode)

        show_results = has_result or is_analyzing
        self._empty_state.setVisible(not show_results)
        for section in self._sections:
            if section not in self._detail_sections or not self._presentation_mode:
                section.setVisible(show_results)

        if is_analyzing and not has_result:
            for section in self._sections:
                section.set_loading(True)
            self._toolbar_enabled(False)
            return

        self._toolbar_enabled(has_result and not self._presentation_mode)
        for section in self._sections:
            section.set_loading(False)

        self._narrative.set_text(narrative_text)
        self._short_caption.set_text(short_caption_text)
        self._executive.set_text(executive_text)
        self._scene_description.set_text(scene_description_text)
        self._caption.set_text(caption)
        self._scene_summary.set_text(scene_summary)
        self._objects.set_text(objects_text)
        self._attributes.set_text(attributes_text)
        self._relationships.set_text(relationships_text)
        self._activities.set_text(activities_text)
        self._environment.set_text(environment_text)
        self._image_quality.set_text(
            image_quality_text if has_result else "No image quality report available yet."
        )
        self._reasoning.set_text(reasoning_text)
        self._confidence.set_text(confidence_text)
        self._quality.set_text(quality_text if has_result else "No quality report available yet.")
        self._metrics.set_text(metrics_text if has_result else "No execution metrics available yet.")
        self._apply_filter(self._search.text())

    def _toolbar_enabled(self, enabled: bool) -> None:
        self._copy_all_button.setEnabled(enabled)
        self._expand_all_button.setEnabled(enabled)
        self._collapse_all_button.setEnabled(enabled)
        self._search.setEnabled(enabled)

    def _apply_filter(self, query: str) -> None:
        if not self._has_result and not self._is_analyzing:
            return
        for section in self._sections:
            if self._presentation_mode and section in self._detail_sections:
                continue
            section.set_visible_for_filter(section.matches_filter(query))

    def _expand_all(self) -> None:
        for section in self._sections:
            if section.isVisible():
                section.expand()

    def _collapse_all(self) -> None:
        for section in self._sections:
            if section.isVisible():
                section.collapse()

    def _copy_all(self) -> None:
        chunks = [
            f"{section.section_title}\n{section.plain_text}"
            for section in self._sections
            if section.isVisible() and section.plain_text
        ]
        if chunks:
            QGuiApplication.clipboard().setText("\n\n".join(chunks))

    def _on_section_copied(self, title: str) -> None:
        self._status_label.setText(f"Copied {title} to clipboard")
