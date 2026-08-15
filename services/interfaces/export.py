"""Export service interface."""

from pathlib import Path
from typing import Protocol

from core.contracts.pipeline import PipelineResult


class IExportService(Protocol):
    """Export pipeline results."""

    def export(self, result: PipelineResult, export_format: str, path: Path) -> None:
        ...
