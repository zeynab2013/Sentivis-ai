"""UI-facing runtime status provider."""

from __future__ import annotations

from dataclasses import dataclass

from services.runtime.cache_maintenance import CacheMaintenanceService, CacheSizeReport
from services.runtime.model_record import ModelRecord
from services.runtime.model_registry import CentralModelRegistry
from services.runtime.self_test import SelfTestReport, SelfTestRunner


@dataclass(frozen=True)
class ModelStatusView:
    """Read-only model status for UI adapters."""

    kind: str
    display_name: str
    identifier: str
    version: str
    provider: str
    supported_tasks: tuple[str, ...]
    file_location: str | None
    device_compatibility: tuple[str, ...]
    runtime_status: str
    integrity_status: str
    last_validation_time: str | None
    validation_detail: str


class RuntimeStatusProvider:
    """Exposes model status, cache metrics, and self-test results without UI coupling."""

    def __init__(
        self,
        model_registry: CentralModelRegistry,
        cache_maintenance: CacheMaintenanceService,
        self_test_runner: SelfTestRunner,
    ) -> None:
        self._model_registry = model_registry
        self._cache_maintenance = cache_maintenance
        self._self_test_runner = self_test_runner
        self._last_self_test: SelfTestReport | None = None

    @property
    def health_score(self) -> int:
        return self.latest_self_test().health_score

    def latest_self_test(self) -> SelfTestReport:
        if self._last_self_test is None:
            self._last_self_test = self._self_test_runner.run()
        return self._last_self_test

    def refresh(self) -> SelfTestReport:
        self._model_registry.refresh()
        self._last_self_test = self._self_test_runner.run()
        return self._last_self_test

    def cache_report(self) -> CacheSizeReport:
        return self._cache_maintenance.report_size()

    def model_statuses(self) -> tuple[ModelStatusView, ...]:
        return tuple(self._to_view(record) for record in self._model_registry.records)

    def get_model_status(self, kind_name: str) -> ModelStatusView | None:
        for view in self.model_statuses():
            if view.kind == kind_name.lower():
                return view
        return None

    @staticmethod
    def _to_view(record: ModelRecord) -> ModelStatusView:
        return ModelStatusView(
            kind=record.kind.name.lower(),
            display_name=record.display_name,
            identifier=record.identifier,
            version=record.version,
            provider=record.provider,
            supported_tasks=record.supported_tasks,
            file_location=str(record.file_location) if record.file_location else None,
            device_compatibility=record.device_compatibility,
            runtime_status=record.runtime_status.value,
            integrity_status=record.integrity_status.value,
            last_validation_time=(
                record.last_validation_time.isoformat() if record.last_validation_time else None
            ),
            validation_detail=record.validation_detail,
        )
