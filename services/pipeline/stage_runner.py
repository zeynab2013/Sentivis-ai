"""Pipeline stage execution with lifecycle management."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from core.constants.pipeline_stages import PipelineStage
from core.exceptions.base import SentivisError
from core.exceptions.service import CancelledError, OrchestrationError
from core.logging import get_logger
from services.interfaces.cancellation import ICancellationToken
from services.interfaces.progress import IProgressReporter
from services.memory.managed_resources import ResourceScope
from services.memory.memory_manager import MemoryManager
from services.pipeline.pipeline_guard import PipelineGuard

if TYPE_CHECKING:
    from services.models.model_manager import ModelManager
    from services.pipeline.metrics_collector import PipelineMetricsCollector

logger = get_logger(__name__)

TOut = TypeVar("TOut")

_GPU_STAGES = frozenset(
    {
        PipelineStage.YOLO_DETECTION,
        PipelineStage.BLIP_UNDERSTANDING,
        PipelineStage.GEMMA_REASONING,
    }
)


class StageRunner:
    """Executes individual pipeline stages with timing, logging, and cleanup."""

    def __init__(
        self,
        progress: IProgressReporter,
        cancellation: ICancellationToken,
        memory_manager: MemoryManager | None = None,
        pipeline_guard: PipelineGuard | None = None,
        metrics_collector: PipelineMetricsCollector | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self._progress = progress
        self._cancellation = cancellation
        self._memory = memory_manager
        self._guard = pipeline_guard
        self._metrics = metrics_collector
        self._model_manager = model_manager

    def set_cancellation(self, cancellation: ICancellationToken) -> None:
        """Bind cancellation token for the active pipeline run."""
        self._cancellation = cancellation

    def run(
        self,
        stage: PipelineStage,
        percent: float,
        message: str,
        action: Callable[[], TOut],
        *,
        recoverable: bool = False,
        fallback: Callable[[], TOut] | None = None,
        validate_output: bool = True,
    ) -> TOut:
        """Run one stage action with full lifecycle handling."""
        self._cancellation.raise_if_cancelled()
        if self._guard:
            self._guard.before_stage(stage)

        scope = ResourceScope()
        self._progress.emit(stage, percent, message)
        logger.info("Stage %s started: %s", stage.name, message)
        if self._memory:
            self._memory.log_snapshot(f"{stage.name} before")

        stage_started = time.perf_counter()
        try:
            result = self._run_action(stage, action, recoverable, fallback)
            self._cancellation.raise_if_cancelled()
            if validate_output and self._guard and result is not None:
                self._guard.after_stage(stage, result)
            return result
        except CancelledError:
            raise
        except SentivisError as exc:
            if recoverable and fallback is not None:
                logger.warning(
                    "Stage %s recoverable failure: %s",
                    stage.name,
                    exc.developer_detail,
                )
                if self._metrics:
                    self._metrics.record_recovery()
                return fallback()
            raise
        except Exception as exc:
            raise OrchestrationError(
                "Analysis could not be completed.",
                f"Stage {stage.name} failed: {exc}",
                stage=stage,
                recoverable=False,
            ) from exc
        finally:
            scope.dispose_all()
            if stage in _GPU_STAGES and self._memory:
                self._memory.clear_gpu_cache()
                if self._model_manager and self._metrics:
                    cycle = self._model_manager.consume_last_cycle()
                    if cycle:
                        self._metrics.record_model_timing(
                            cycle.kind,
                            cycle.load_ms,
                            cycle.unload_ms,
                            cycle.gpu_released,
                        )
            else:
                gc.collect()
            if self._memory:
                self._memory.log_snapshot(f"{stage.name} after")
            duration_ms = max((time.perf_counter() - stage_started) * 1000.0, 0.1)
            if self._metrics:
                self._metrics.record_stage(stage, duration_ms)
            logger.info("Stage %s finished in %.1f ms", stage.name, duration_ms)

    def _run_action(
        self,
        stage: PipelineStage,
        action: Callable[[], TOut],
        recoverable: bool,
        fallback: Callable[[], TOut] | None,
    ) -> TOut:
        try:
            return action()
        except SentivisError as exc:
            if exc.recoverable and stage in _GPU_STAGES and self._memory is not None:
                logger.warning("Retrying stage %s once after cleanup", stage.name)
                if self._metrics:
                    self._metrics.record_recovery()
                self._memory.recover_from_oom()
                return action()
            if recoverable and fallback is not None:
                if self._metrics:
                    self._metrics.record_recovery()
                return fallback()
            raise
