"""Settings view model interface."""

from typing import Protocol


class ISettingsViewModel(Protocol):
    """UI-facing settings presentation contract."""

    @property
    def app_name(self) -> str:
        ...

    @property
    def app_version(self) -> str:
        ...

    @property
    def theme_name(self) -> str:
        ...

    def set_theme(self, theme_name: str, window: object) -> None:
        ...
