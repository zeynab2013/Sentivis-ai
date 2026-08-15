"""History presentation state."""

from PySide6.QtCore import QObject, Signal

from ui.interfaces.history_view_model import HistoryEntry


class HistoryViewModel(QObject):
    """Session analysis history."""

    history_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[HistoryEntry] = []

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries)

    def add_entry(
        self,
        image_name: str,
        caption_preview: str,
        *,
        analyzed_at: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        self._entries.append(
            HistoryEntry(
                image_name,
                caption_preview,
                analyzed_at=analyzed_at,
                duration_ms=duration_ms,
            )
        )
        self.history_changed.emit()
