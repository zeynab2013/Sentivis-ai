"""History view model interface."""

from typing import Protocol


class HistoryEntry:
    """One history list item."""

    def __init__(
        self,
        image_name: str,
        caption_preview: str,
        *,
        analyzed_at: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        self.image_name = image_name
        self.caption_preview = caption_preview
        self.analyzed_at = analyzed_at
        self.duration_ms = duration_ms


class IHistoryViewModel(Protocol):
    """Session history state."""

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        ...

    def add_entry(self, image_name: str, caption_preview: str) -> None:
        ...
