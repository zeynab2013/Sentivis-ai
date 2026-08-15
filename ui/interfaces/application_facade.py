"""Application facade for UI layer."""

from typing import Protocol

from core.config.app_config import AppConfig
from core.config.theme_config import ThemeConfig
from ui.interfaces.export_view_model import IExportViewModel
from ui.interfaces.history_view_model import IHistoryViewModel
from ui.interfaces.pipeline_view_model import IPipelineViewModel
from ui.interfaces.settings_view_model import ISettingsViewModel
from ui.themes.theme_manager import ThemeManager


class IApplicationFacade(Protocol):
    """UI-facing application entry without infrastructure imports."""

    @property
    def window_title(self) -> str:
        ...

    @property
    def pipeline_view_model(self) -> IPipelineViewModel:
        ...

    @property
    def export_view_model(self) -> IExportViewModel:
        ...

    @property
    def history_view_model(self) -> IHistoryViewModel:
        ...

    @property
    def theme_config(self) -> ThemeConfig:
        ...

    @property
    def app_config(self) -> AppConfig:
        ...

    @property
    def theme_manager(self) -> ThemeManager:
        ...

    @property
    def settings_view_model(self) -> ISettingsViewModel:
        ...

