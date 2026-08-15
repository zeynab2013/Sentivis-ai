"""Download progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DownloadState(str, Enum):  # noqa: UP042
    """State of an individual model download."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """Progress update for one model download."""

    model_name: str
    state: DownloadState
    bytes_downloaded: int = 0
    total_bytes: int | None = None
    message: str = ""
    attempt: int = 1

    @property
    def percent(self) -> float | None:
        if self.total_bytes is None or self.total_bytes <= 0:
            return None
        return min(100.0, (self.bytes_downloaded / self.total_bytes) * 100.0)
