"""Design-system scroll panel."""

from PySide6.QtWidgets import QScrollArea, QWidget


class SentivisScrollPanel(QScrollArea):
    """Scrollable panel with consistent chrome."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

    def set_content(self, widget: QWidget) -> None:
        self.setWidget(widget)
