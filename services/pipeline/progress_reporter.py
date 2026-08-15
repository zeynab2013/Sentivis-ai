"""Pipeline progress reporting."""

from collections.abc import Callable

from core.constants.pipeline_stages import PipelineStage
from core.contracts.pipeline import StageProgress


class ProgressReporter:
    """Forwards pipeline progress to registered listeners."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[StageProgress], None]] = []

    def subscribe(self, listener: Callable[[StageProgress], None]) -> None:
        self._listeners.append(listener)

    def emit(
        self,
        stage: PipelineStage,
        percent: float,
        message: str,
        device: str = "cpu",
    ) -> None:
        progress = StageProgress(stage=stage, percent=percent, message=message, device=device)
        for listener in self._listeners:
            listener(progress)
