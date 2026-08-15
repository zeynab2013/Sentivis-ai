"""Non-blocking notification toasts."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.models.operation_status import OperationStatus


class _NotificationToast(QWidget):
    """Single auto-dismissing toast."""

    def __init__(
        self,
        message: str,
        level: str,
        duration_ms: int,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "NotificationToast")
        self.setProperty("level", level)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QLabel(message)
        label.setWordWrap(True)
        label.setProperty("class", "NotificationMessage")
        layout.addWidget(label)
        QTimer.singleShot(duration_ms, self.deleteLater)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        self.style().unpolish(self)
        self.style().polish(self)


class NotificationCenter(QWidget):
    """Stacked toast notifications anchored to the parent window."""

    _DURATIONS = {
        "info": 4000,
        "success": 4500,
        "warning": 7000,
        "error": 9000,
    }

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.raise_()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def show_info(self, message: str) -> None:
        self._show(message, "info")

    def show_success(self, message: str) -> None:
        self._show(message, "success")

    def show_warning(self, message: str) -> None:
        self._show(message, "warning")

    def show_error(self, message: str) -> None:
        self._show(message, "error")

    def show_status(self, status: OperationStatus, message: str) -> None:
        mapping = {
            OperationStatus.COMPLETED: "success",
            OperationStatus.WARNING: "warning",
            OperationStatus.FAILED: "error",
            OperationStatus.RECOVERING: "warning",
        }
        self._show(message, mapping.get(status, "info"))

    def _show(self, message: str, level: str) -> None:
        toast = _NotificationToast(message, level, self._DURATIONS[level], self)
        self._layout.insertWidget(self._layout.count() - 1, toast)
        toast.show()
        toast.style().unpolish(toast)
        toast.style().polish(toast)
        self.raise_()
