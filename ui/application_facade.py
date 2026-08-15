"""Application facade implementation."""

from core.config.app_config import AppConfig
from core.config.theme_config import ThemeConfig
from ui.themes.theme_manager import ThemeManager
from ui.view_models.export_view_model import ExportViewModel
from ui.view_models.history_view_model import HistoryViewModel
from ui.view_models.pipeline_view_model import PipelineViewModel
from ui.view_models.settings_view_model import SettingsViewModel


class ApplicationFacade:
    """Provides ViewModels to the UI without exposing services."""

    def __init__(
        self,
        app_config: AppConfig,
        theme_config: ThemeConfig,
        theme_manager: ThemeManager,
        pipeline_view_model: PipelineViewModel,
        export_view_model: ExportViewModel,
        history_view_model: HistoryViewModel,
        settings_view_model: SettingsViewModel,
    ) -> None:
        self._app_config = app_config
        self._theme_config = theme_config
        self._theme_manager = theme_manager
        self._pipeline_view_model = pipeline_view_model
        self._export_view_model = export_view_model
        self._history_view_model = history_view_model
        self._settings_view_model = settings_view_model

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @property
    def window_title(self) -> str:
        return f"{self._app_config.app_name} — SEE. UNDERSTAND. INSPIRE."

    @property
    def theme_config(self) -> ThemeConfig:
        return self._theme_config

    @property
    def theme_manager(self) -> ThemeManager:
        return self._theme_manager

    @property
    def pipeline_view_model(self) -> PipelineViewModel:
        return self._pipeline_view_model

    @property
    def export_view_model(self) -> ExportViewModel:
        return self._export_view_model

    @property
    def history_view_model(self) -> HistoryViewModel:
        return self._history_view_model

    @property
    def settings_view_model(self) -> SettingsViewModel:
        return self._settings_view_model
