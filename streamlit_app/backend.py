"""Streamlit backend adapter type (no startup side effects)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.container import ApplicationContext
from core.config.app_config import AppConfig
from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult, StageProgress
from services.export.export_manager import ExportManager
from services.interfaces.pipeline import IPipelineOrchestrator
from services.interfaces.progress import IProgressReporter


@dataclass(frozen=True)
class StreamlitBackend:
    """Thin adapter over the existing DI stack for Streamlit."""

    context: ApplicationContext
    orchestrator: IPipelineOrchestrator
    progress: IProgressReporter
    export_manager: ExportManager

    @property
    def app_config(self) -> AppConfig:
        return self.context.facade.app_config

    def analyze(
        self,
        image_path: Path,
        *,
        competition_mode: bool = False,
        enable_enhancement: bool = True,
        enable_super_resolution: bool = False,
        enable_sam2: bool = True,
        enable_gemma: bool = True,
        on_progress: Callable[[StageProgress], None] | None = None,
    ) -> PipelineResult:
        if on_progress is not None:
            self.progress.subscribe(on_progress)
        try:
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
            return self.orchestrator.analyze(request)
        finally:
            if on_progress is not None and hasattr(self.progress, "_listeners"):
                listeners = self.progress._listeners  # noqa: SLF001
                if on_progress in listeners:
                    listeners.remove(on_progress)

    def cancel(self) -> None:
        self.orchestrator.cancel()

    def shutdown(self) -> None:
        self.context.model_manager.release_all()
        self.context.memory_manager.clear_gpu_cache()
