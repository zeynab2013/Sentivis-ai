"""Stage runner interface."""

from collections.abc import Callable
from typing import Protocol, TypeVar

from core.constants.pipeline_stages import PipelineStage
from services.interfaces.cancellation import ICancellationToken

TOut = TypeVar("TOut")


class IStageRunner(Protocol):
    """Execute a single pipeline stage with lifecycle management."""

    def set_cancellation(self, cancellation: ICancellationToken) -> None:
        ...

    def run(
        self,
        stage: PipelineStage,
        percent: float,
        message: str,
        action: Callable[[], TOut],
        *,
        recoverable: bool = False,
        fallback: Callable[[], TOut] | None = None,
    ) -> TOut:
        ...
