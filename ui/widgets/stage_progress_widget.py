"""Per-stage pipeline progress display."""

from PySide6.QtWidgets import QLabel, QVBoxLayout

from ui.components.card import SentivisCard
from ui.models.pipeline_ui_state import StageDisplayItem, StageStatus

_STATUS_LABELS = {
    StageStatus.PENDING: ("○", "stage-pending"),
    StageStatus.RUNNING: ("●", "stage-running"),
    StageStatus.COMPLETED: ("✓", "stage-completed"),
    StageStatus.FAILED: ("✕", "stage-failed"),
    StageStatus.SKIPPED: ("–", "stage-skipped"),
}


class StageProgressWidget(SentivisCard):
    """Lists pipeline stages with status and duration."""

    def __init__(self) -> None:
        super().__init__()
        layout = self.content_layout
        title = QLabel("Pipeline Progress")
        title.setProperty("class", "SectionTitle")
        layout.addWidget(title)
        self._rows_layout = QVBoxLayout()
        layout.addLayout(self._rows_layout)
        self._status_hint = QLabel("")
        self._status_hint.setProperty("class", "StatusLabel")
        self._status_hint.setWordWrap(True)
        layout.addWidget(self._status_hint)
        self._row_labels: list[QLabel] = []

    def bind(
        self,
        stages: tuple[StageDisplayItem, ...],
        *,
        is_analyzing: bool = False,
        stage_label: str = "",
    ) -> None:
        while len(self._row_labels) < len(stages):
            label = QLabel()
            label.setProperty("class", "StageRow")
            self._rows_layout.addWidget(label)
            self._row_labels.append(label)
        while len(self._row_labels) > len(stages):
            widget = self._row_labels.pop()
            widget.deleteLater()

        for label, item in zip(self._row_labels, stages, strict=True):
            icon, css_class = _STATUS_LABELS[item.status]
            duration = ""
            if item.duration_ms is not None and item.status in {
                StageStatus.COMPLETED,
                StageStatus.FAILED,
            }:
                duration = f" — {item.duration_ms:.0f} ms"
            label.setText(f"{icon}  {item.label}{duration}")
            label.setProperty("class", css_class)
            label.style().unpolish(label)
            label.style().polish(label)

        if is_analyzing and stage_label:
            self._status_hint.setText(f"Current stage: {stage_label}")
            self._status_hint.show()
        elif is_analyzing:
            self._status_hint.setText("Analysis in progress…")
            self._status_hint.show()
        else:
            self._status_hint.hide()
