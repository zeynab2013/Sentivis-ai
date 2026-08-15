"""Design-system progress bar."""

from PySide6.QtWidgets import QProgressBar


class SentivisProgressBar(QProgressBar):
    """Token-styled progress indicator."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setRange(0, 100)
        self.setTextVisible(True)
