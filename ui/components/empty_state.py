"""Reusable empty-state presentation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class EmptyStateWidget(QWidget):
    """Guides the user when a panel has no content yet."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        action_hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._title = QLabel(title)
        self._title.setProperty("class", "EmptyStateTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._message = QLabel(message)
        self._message.setProperty("class", "EmptyStateMessage")
        self._message.setWordWrap(True)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title)
        layout.addWidget(self._message)

        self._hint = QLabel(action_hint)
        self._hint.setProperty("class", "EmptyStateHint")
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if action_hint:
            layout.addWidget(self._hint)
        else:
            self._hint.hide()

    def set_content(self, title: str, message: str, action_hint: str = "") -> None:
        self._title.setText(title)
        self._message.setText(message)
        if action_hint:
            self._hint.setText(action_hint)
            self._hint.show()
        else:
            self._hint.hide()
