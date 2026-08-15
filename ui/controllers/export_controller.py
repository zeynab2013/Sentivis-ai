"""Export UI controller."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.config.app_config import AppConfig
from core.contracts.pipeline import PipelineResult
from core.exceptions.base import SentivisError
from services.export.export_manager import ExportManager


class ExportController(QObject):
    """Handles export requests from the UI."""

    export_succeeded = Signal(str)
    export_failed = Signal(str)

    def __init__(self, export_manager: ExportManager, app_config: AppConfig) -> None:
        super().__init__()
        self._export_manager = export_manager
        self._exports_dir = app_config.paths.exports_dir

    @property
    def exports_directory(self) -> Path:
        return self._exports_dir

    def preview_export_path(self, result: PipelineResult | None, export_format: str) -> Path | None:
        if result is None:
            return None
        stem = result.request.image_path.stem
        extension = export_format if export_format != "image" else result.request.image_path.suffix.lstrip(".")
        return self._exports_dir / f"{stem}_sentivis.{extension}"

    def export_result(self, result: PipelineResult, export_format: str) -> None:
        path = self.preview_export_path(result, export_format)
        if path is None:
            return
        try:
            self._export_manager.export(result, export_format, path)
            self.export_succeeded.emit(str(path))
        except SentivisError as exc:
            self.export_failed.emit(exc.user_message)
