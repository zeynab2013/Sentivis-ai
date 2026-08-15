"""Settings UI controller."""

from PySide6.QtCore import QObject

from core.config.app_config import AppConfig
from core.config.theme_config import ThemeConfig


class SettingsController(QObject):
    """Exposes settings and theme configuration to the UI."""

    def __init__(self, app_config: AppConfig, theme_config: ThemeConfig) -> None:
        super().__init__()
        self._app_config = app_config
        self._theme_config = theme_config

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @property
    def theme_config(self) -> ThemeConfig:
        return self._theme_config
