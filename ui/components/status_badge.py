"""Design-system status badge."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from ui.models.operation_status import OperationStatus, operation_status_label


class StatusBadge(QLabel):
    """Accessible status indicator with text and token styling."""

    _STATUS_PREFIX = {
        OperationStatus.READY: "●",
        OperationStatus.LOADING: "◌",
        OperationStatus.RUNNING: "▶",
        OperationStatus.WAITING: "⏳",
        OperationStatus.RECOVERING: "↻",
        OperationStatus.COMPLETED: "✓",
        OperationStatus.WARNING: "!",
        OperationStatus.FAILED: "✕",
    }

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status(OperationStatus.READY)

    def set_status(self, status: OperationStatus, detail: str = "") -> None:
        prefix = self._STATUS_PREFIX.get(status, "●")
        label = operation_status_label(status)
        text = f"{prefix} {label}"
        if detail:
            text = f"{text} — {detail}"
        self.setText(text)
        self.setProperty("status", status.value)
        self.setToolTip(f"Status: {label}" + (f"\n{detail}" if detail else ""))
        self.style().unpolish(self)
        self.style().polish(self)
