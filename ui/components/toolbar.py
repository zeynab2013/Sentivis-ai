"""Design-system toolbar."""

from PySide6.QtWidgets import QHBoxLayout, QWidget


class SentivisToolbar(QWidget):
    """Horizontal toolbar for grouped actions."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

    @property
    def toolbar_layout(self) -> QHBoxLayout:
        return self._layout

    def add_widget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def add_stretch(self) -> None:
        self._layout.addStretch()
