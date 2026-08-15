"""Design-system button."""

from PySide6.QtWidgets import QPushButton, QSizePolicy


class SentivisButton(QPushButton):
    """Token-styled button with primary, secondary, or danger variants."""

    def __init__(self, text: str, *, variant: str = "primary", parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setCursor(self.cursor())
        # Prefer horizontal growth over cramped clipping for multilingual labels.
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(36)
        self.setStyleSheet("QPushButton { padding: 8px 14px; text-align: center; }")

    def set_variant(self, variant: str) -> None:
        self.setProperty("variant", variant)
        self.style().unpolish(self)
        self.style().polish(self)
