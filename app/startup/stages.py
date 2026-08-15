"""Startup stage definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter


class StartupStage(Enum):
    """Ordered startup stages reported to the user and logs."""

    ENVIRONMENT_VALIDATION = "environment_validation"
    CONFIGURATION_LOADING = "configuration_loading"
    DEPENDENCY_VERIFICATION = "dependency_verification"
    MODEL_DISCOVERY = "model_discovery"
    PLUGIN_DISCOVERY = "plugin_discovery"
    RESOURCE_INITIALIZATION = "resource_initialization"
    THEME_INITIALIZATION = "theme_initialization"
    APPLICATION_READY = "application_ready"


@dataclass(frozen=True)
class StageProgress:
    """Result of one startup stage."""

    stage: StartupStage
    message: str
    duration_ms: float
    warnings: tuple[str, ...] = ()
    recoverable: bool = True


@dataclass
class StartupReport:
    """Aggregated startup progress and issues."""

    stages: list[StageProgress] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_stage(
        self,
        stage: StartupStage,
        message: str,
        started_at: float,
        *,
        warnings: tuple[str, ...] = (),
    ) -> None:
        duration_ms = (perf_counter() - started_at) * 1000.0
        self.stages.append(
            StageProgress(stage=stage, message=message, duration_ms=duration_ms, warnings=warnings)
        )
        self.warnings.extend(warnings)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def ready(self) -> bool:
        return not self.errors
