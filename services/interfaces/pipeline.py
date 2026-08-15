"""Pipeline orchestrator interface."""

from typing import Protocol

from core.contracts.pipeline import PipelineRequest, PipelineResult


class IPipelineOrchestrator(Protocol):
    """Canonical pipeline coordination."""

    def analyze(self, request: PipelineRequest) -> PipelineResult:
        ...

    def cancel(self) -> None:
        ...
