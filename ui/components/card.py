"""Design-system card container."""

from PySide6.QtWidgets import QVBoxLayout, QWidget


class SentivisCard(QWidget):
    """Elevated surface container using token-driven QSS."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setProperty("class", "Card")
        self._layout = QVBoxLayout(self)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._layout
