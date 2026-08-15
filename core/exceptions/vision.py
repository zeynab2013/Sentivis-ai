"""Vision module exceptions."""

from core.constants.pipeline_stages import PipelineStage
from core.exceptions.base import SentivisError


class ValidationError(SentivisError):
    """Raised when an image fails validation."""

    def __init__(self, user_message: str, developer_detail: str) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=False,
            stage=PipelineStage.VALIDATION,
        )


class DetectionError(SentivisError):
    """Raised when object detection fails."""

    def __init__(
        self,
        user_message: str,
        developer_detail: str,
        *,
        recoverable: bool = False,
    ) -> None:
        super().__init__(
            user_message,
            developer_detail,
            recoverable=recoverable,
            stage=PipelineStage.YOLO_DETECTION,
        )
