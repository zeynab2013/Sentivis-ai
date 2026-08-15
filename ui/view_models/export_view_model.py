"""Export presentation state with destination preview and progress."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ui.controllers.export_controller import ExportController
from ui.view_models.pipeline_view_model import PipelineViewModel


class ExportViewModel(QObject):
    """Manages export UI state, confirmation, and progress signaling."""

    export_status_changed = Signal(str)
    export_started = Signal(str)
    export_finished = Signal()

    SUPPORTED_FORMATS: tuple[str, ...] = ("json", "txt", "md", "pdf")

    def __init__(
        self,
        controller: ExportController,
        pipeline_view_model: PipelineViewModel,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._pipeline_view_model = pipeline_view_model
        self._last_export_path = ""
        controller.export_succeeded.connect(self._on_success)
        controller.export_failed.connect(self._on_failure)

    @property
    def is_export_enabled(self) -> bool:
        return (
            self._pipeline_view_model.current_result is not None
            and not self._pipeline_view_model.exporting
        )

    @property
    def last_export_path(self) -> str:
        return self._last_export_path

    @property
    def exports_directory(self) -> Path:
        return self._controller.exports_directory

    def preview_path(self, export_format: str) -> Path | None:
        return self._controller.preview_export_path(
            self._pipeline_view_model.current_result,
            export_format,
        )

    def export_json(self, *, confirm_overwrite: bool = True) -> None:
        self._export("json", confirm_overwrite=confirm_overwrite)

    def export_txt(self, *, confirm_overwrite: bool = True) -> None:
        self._export("txt", confirm_overwrite=confirm_overwrite)

    def export_md(self, *, confirm_overwrite: bool = True) -> None:
        self._export("md", confirm_overwrite=confirm_overwrite)

    def export_pdf(self, *, confirm_overwrite: bool = True) -> None:
        self._export("pdf", confirm_overwrite=confirm_overwrite)

    def _export(self, export_format: str, *, confirm_overwrite: bool) -> None:
        result = self._pipeline_view_model.current_result
        if result is None:
            return
        preview = self._controller.preview_export_path(result, export_format)
        if preview is None:
            return
        if confirm_overwrite and preview.exists():
            self.export_status_changed.emit(f"confirm_overwrite:{preview}")
            return
        self._pipeline_view_model.set_exporting(True)
        self.export_started.emit(str(preview))
        self._controller.export_result(result, export_format)

    def confirm_export(self, export_format: str) -> None:
        self._export(export_format, confirm_overwrite=False)

    def _on_success(self, path: str) -> None:
        self._pipeline_view_model.set_exporting(False)
        self._last_export_path = path
        self.export_finished.emit()
        self.export_status_changed.emit(path)

    def _on_failure(self, message: str) -> None:
        self._pipeline_view_model.set_exporting(False)
        self.export_finished.emit()
        self.export_status_changed.emit(message)
