"""Export view model interface."""

from typing import Protocol


class IExportViewModel(Protocol):
    """Bindable export state."""

    @property
    def is_export_enabled(self) -> bool:
        ...

    @property
    def last_export_path(self) -> str:
        ...

    def export_json(self) -> None:
        ...

    def export_txt(self) -> None:
        ...

    def export_pdf(self) -> None:
        ...
