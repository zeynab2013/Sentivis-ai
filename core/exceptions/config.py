"""Configuration exceptions."""

from core.exceptions.base import SentivisError


class ConfigurationError(SentivisError):
    """Raised when configuration fails validation."""

    def __init__(self, developer_detail: str) -> None:
        super().__init__(
            "Application configuration is invalid. See log for details.",
            developer_detail,
            recoverable=False,
        )
