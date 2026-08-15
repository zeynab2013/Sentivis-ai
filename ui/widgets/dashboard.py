"""Right-side analysis dashboard."""

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ui.components.scroll_panel import SentivisScrollPanel
from ui.widgets.export_panel import ExportPanelWidget
from ui.widgets.results_panel import ResultsPanelWidget
from ui.widgets.stage_progress_widget import StageProgressWidget


class AnalysisDashboardWidget(QWidget):
    """Pipeline progress, results, and export actions."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(340)
        outer = QVBoxLayout(self)
        scroll = SentivisScrollPanel()

        container = QWidget()
        layout = QVBoxLayout(container)
        self.stage_progress = StageProgressWidget()
        self.results_panel = ResultsPanelWidget()
        self.export_panel = ExportPanelWidget()
        layout.addWidget(self.stage_progress)
        layout.addWidget(self.results_panel, stretch=1)
        layout.addWidget(self.export_panel)
        scroll.set_content(container)
        outer.addWidget(scroll)

        self._presentation_mode = False

    def set_presentation_mode(self, enabled: bool) -> None:
        self._presentation_mode = enabled
        self.stage_progress.setVisible(not enabled)
        self.results_panel.set_presentation_mode(enabled)
        self.export_panel.set_presentation_mode(enabled)
        self.setMinimumWidth(300 if enabled else 340)

    def retranslate_ui(self) -> None:
        self.results_panel.retranslate_ui()
        self.export_panel.retranslate_ui()
