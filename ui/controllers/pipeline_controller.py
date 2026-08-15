"""Pipeline UI controller."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult, StageProgress
from services.interfaces.pipeline import IPipelineOrchestrator
from services.interfaces.progress import IProgressReporter
from ui.workers.pipeline_worker import PipelineWorker


class PipelineController(QObject):
    """Bridges UI actions to pipeline orchestration."""

    progress_changed = Signal(object)
    analysis_completed = Signal(object)
    analysis_failed = Signal(str)

    def __init__(
        self,
        orchestrator: IPipelineOrchestrator,
        progress: IProgressReporter,
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._progress = progress
        self._worker: PipelineWorker | None = None
        self._progress.subscribe(self._forward_progress)

    def analyze_image(
        self,
        image_path: Path,
        enable_gemma: bool = True,
        *,
        competition_mode: bool = False,
        enable_enhancement: bool = True,
        enable_super_resolution: bool = False,
        enable_sam2: bool = True,
    ) -> None:
        if self._worker and self._worker.isRunning():
            return
        request = PipelineRequest(
            image_path=image_path,
            options=AnalysisOptions(
                enable_gemma=enable_gemma,
                competition_mode=competition_mode,
                enable_enhancement=enable_enhancement,
                enable_super_resolution=enable_super_resolution,
                enable_sam2=enable_sam2,
            ),
        )
        self._worker = PipelineWorker(self._orchestrator, request)
        self._worker.progress.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self.analysis_failed.emit)
        self._worker.start()

    def cancel(self) -> None:
        self._orchestrator.cancel()

    def _forward_progress(self, event: StageProgress) -> None:
        self.progress_changed.emit(event)

    def _on_finished(self, result: object) -> None:
        if isinstance(result, PipelineResult):
            self.analysis_completed.emit(result)
