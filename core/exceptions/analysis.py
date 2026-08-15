"""Analysis module exceptions."""

from core.constants.pipeline_stages import PipelineStage
from core.exceptions.base import SentivisError


class AnalysisError(SentivisError):
    """Raised when scene analysis fails."""

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
