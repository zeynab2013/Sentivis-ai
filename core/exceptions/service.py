"""Service layer exceptions."""

from core.constants.pipeline_stages import PipelineStage
from core.exceptions.base import SentivisError


class OrchestrationError(SentivisError):
    """Raised when pipeline orchestration fails."""

    def __init__(
        self,
        user_message: str,
        developer_detail: str,
        *,
        stage: PipelineStage | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=recoverable,
            stage=stage,
        )


class ExportError(SentivisError):
    """Raised when export fails."""

    def __init__(self, user_message: str, developer_detail: str) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=True,
            stage=PipelineStage.EXPORT,
        )


class CancelledError(SentivisError):
    """Raised when pipeline execution is cancelled."""

    def __init__(self, developer_detail: str = "Pipeline cancelled by user") -> None:
        super().__init__(
            "Analysis was cancelled.",
            developer_detail,
            recoverable=False,
            stage=PipelineStage.EXPORT,
        )


class PipelineTimeoutError(SentivisError):
    """Raised when pipeline execution exceeds configured timeout."""

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            "Analysis took too long and was stopped.",
            f"Pipeline exceeded timeout of {timeout_seconds}s",
            recoverable=False,
        )

