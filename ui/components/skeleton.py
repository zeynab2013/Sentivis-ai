"""Loading skeleton placeholders."""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SkeletonBlock(QWidget):
    """Placeholder block shown while content is loading."""

    def __init__(self, lines: int = 3, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for index in range(lines):
            line = QLabel("")
            line.setFixedHeight(12)
            line.setProperty("class", "SkeletonLine")
            if index == lines - 1:
                line.setMaximumWidth(180)
            layout.addWidget(line)
