"""Pipeline view model interface."""

from pathlib import Path
from typing import Protocol


class IPipelineViewModel(Protocol):
    """Bindable pipeline state for UI widgets."""

    @property
    def progress_percent(self) -> float:
        ...

    @property
    def stage_label(self) -> str:
        ...

    @property
    def status_message(self) -> str:
        ...

    @property
    def caption_text(self) -> str:
        ...

    @property
    def scene_summary(self) -> str:
        ...

    @property
    def is_analyze_enabled(self) -> bool:
        ...

    @property
    def image_path(self) -> Path | None:
        ...

    def load_image(self, path: Path) -> None:
        ...

    def start_analysis(self) -> None:
        ...

    def cancel_analysis(self) -> None:
        ...
