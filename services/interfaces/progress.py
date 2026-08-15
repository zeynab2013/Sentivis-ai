"""Progress reporter interface."""

from collections.abc import Callable
from typing import Protocol

from core.constants.pipeline_stages import PipelineStage
from core.contracts.pipeline import StageProgress


class IProgressReporter(Protocol):
    """Emit pipeline progress events."""

    def subscribe(self, listener: Callable[[StageProgress], None]) -> None:
        ...

    def emit(
        self,
        stage: PipelineStage,
        percent: float,
        message: str,
        device: str = "cpu",
    ) -> None:
        ...
