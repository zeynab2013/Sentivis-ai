"""Export actions panel."""

from pathlib import Path

from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from ui.components.button import SentivisButton
from ui.components.card import SentivisCard
from ui.components.empty_state import EmptyStateWidget
from ui.components.progress import SentivisProgressBar
from ui.i18n.translator import tr


class ExportPanelWidget(SentivisCard):
    """Export format actions with destination preview and progress."""

    def __init__(self) -> None:
        super().__init__()
        layout = self.content_layout
        self._heading = QLabel(tr("export.panel.title"))
        self._heading.setProperty("class", "SectionTitle")
        layout.addWidget(self._heading)

        self._stack = QStackedWidget()
        self._empty_state = EmptyStateWidget(
            tr("export.empty.title"),
            tr("export.empty.body"),
            action_hint=tr("export.empty.action"),
        )
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self._destination_label = QLabel(tr("export.destination.empty"))
        self._destination_label.setProperty("class", "StatusLabel")
        self._destination_label.setWordWrap(True)
        content_layout.addWidget(self._destination_label)

        self._formats_label = QLabel(tr("export.formats.supported"))
        self._formats_label.setProperty("class", "StatusLabel")
        content_layout.addWidget(self._formats_label)

        self.export_txt_button = SentivisButton(tr("export.button.txt"), variant="secondary")
        self.export_md_button = SentivisButton(tr("export.button.md"), variant="secondary")
        self.export_json_button = SentivisButton(tr("export.button.json"), variant="secondary")
        self.export_pdf_button = SentivisButton(tr("export.button.pdf"), variant="secondary")
        self.export_txt_button.setToolTip(tr("export.tooltip.txt"))
        self.export_md_button.setToolTip(tr("export.tooltip.md"))
        self.export_json_button.setToolTip(tr("export.tooltip.json"))
        self.export_pdf_button.setToolTip(tr("export.tooltip.pdf"))
        for button in (
            self.export_txt_button,
            self.export_md_button,
            self.export_json_button,
            self.export_pdf_button,
        ):
            content_layout.addWidget(button)

        self._progress = SentivisProgressBar()
        self._progress.hide()
        content_layout.addWidget(self._progress)

        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(content)
        layout.addWidget(self._stack)

    def retranslate_ui(self) -> None:
        self._heading.setText(tr("export.panel.title"))
        self._formats_label.setText(tr("export.formats.supported"))
        self.export_txt_button.setText(tr("export.button.txt"))
        self.export_md_button.setText(tr("export.button.md"))
        self.export_json_button.setText(tr("export.button.json"))
        self.export_pdf_button.setText(tr("export.button.pdf"))
        self.export_txt_button.setToolTip(tr("export.tooltip.txt"))
        self.export_md_button.setToolTip(tr("export.tooltip.md"))
        self.export_json_button.setToolTip(tr("export.tooltip.json"))
        self.export_pdf_button.setToolTip(tr("export.tooltip.pdf"))
        self._empty_state.set_content(
            tr("export.empty.title"),
            tr("export.empty.body"),
            action_hint=tr("export.empty.action"),
        )

    def set_enabled(self, enabled: bool) -> None:
        self._stack.setCurrentIndex(1 if enabled else 0)
        self.export_json_button.setEnabled(enabled)
        self.export_txt_button.setEnabled(enabled)
        self.export_md_button.setEnabled(enabled)
        self.export_pdf_button.setEnabled(enabled)

    def set_destination_preview(self, path: Path | None) -> None:
        if path is None:
            self._destination_label.setText(tr("export.destination.empty"))
        else:
            self._destination_label.setText(tr("export.destination", path=str(path)))

    def set_exporting(self, active: bool) -> None:
        if active:
            self._progress.setRange(0, 0)
            self._progress.show()
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._progress.hide()

    def set_presentation_mode(self, enabled: bool) -> None:
        self.export_json_button.setVisible(not enabled)
        self.export_md_button.setVisible(not enabled)
        self._destination_label.setVisible(not enabled)
