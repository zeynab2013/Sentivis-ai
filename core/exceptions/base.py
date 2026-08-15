"""Base exception for Sentivis AI."""

from core.constants.pipeline_stages import PipelineStage


class SentivisError(Exception):
    """Base exception with user-safe and developer detail messages."""

    def __init__(
        self,
        user_message: str,
        developer_detail: str,
        *,
        recoverable: bool = False,
        stage: PipelineStage | None = None,
    ) -> None:
        """Initialize with separate user and developer messages.

        Args:
            user_message: Safe text for UI display.
            developer_detail: Detailed text for logs.
            recoverable: Whether pipeline may continue with degradation.
            stage: Pipeline stage where the error occurred.
        """
        super().__init__(developer_detail)
        self.user_message = user_message
        self.developer_detail = developer_detail
        self.recoverable = recoverable
        self.stage = stage
