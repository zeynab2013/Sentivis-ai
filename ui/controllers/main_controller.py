"""Main application controller."""

from PySide6.QtCore import QObject

from core.config.app_config import AppConfig
from services.cache.cache_manager import CacheManager
from services.memory.memory_manager import MemoryManager
from services.models.model_manager import ModelManager
from ui.controllers.export_controller import ExportController
from ui.controllers.pipeline_controller import PipelineController
from ui.controllers.settings_controller import SettingsController


class MainController(QObject):
    """Top-level controller coordinating application services."""

    def __init__(
        self,
        app_config: AppConfig,
        pipeline_controller: PipelineController,
        export_controller: ExportController,
        settings_controller: SettingsController,
        model_manager: ModelManager,
        memory_manager: MemoryManager,
        cache_manager: CacheManager,
    ) -> None:
        super().__init__()
        self.app_config = app_config
        self.pipeline = pipeline_controller
        self.export = export_controller
        self.settings = settings_controller
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.cache_manager = cache_manager
