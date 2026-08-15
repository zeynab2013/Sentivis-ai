"""Language module exceptions."""

from core.constants.pipeline_stages import PipelineStage
from core.exceptions.base import SentivisError


class ModelLoadError(SentivisError):
    """Raised when a model fails to load."""

    def __init__(
        self,
        user_message: str,
        developer_detail: str,
        *,
        stage: PipelineStage,
    ) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=False,
            stage=stage,
        )


class InferenceError(SentivisError):
    """Raised when model inference fails."""

    def __init__(
        self,
        user_message: str,
        developer_detail: str,
        *,
        stage: PipelineStage,
        recoverable: bool = True,
    ) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=recoverable,
            stage=stage,
        )
