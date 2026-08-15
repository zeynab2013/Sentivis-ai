"""Model management orchestration service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.constants.model_kinds import ModelKind
from core.logging import get_logger
from model_management.cache import ModelCacheService
from model_management.catalog import PRODUCTION_MODELS, spec_for_kind, total_expected_download_bytes
from model_management.download.manager import DownloadManager, available_disk_bytes
from model_management.download.progress import DownloadProgress
from model_management.hardware import HardwareAssessment, assess_models
from model_management.offline import OfflineReport, is_online, offline_report
from model_management.registry import RegistryEnricher, all_mandatory_validated
from model_management.validation import InstallValidator
from services.runtime.model_record import ModelRecord
from services.runtime.model_registry import CentralModelRegistry
from services.runtime.model_status import InstallationStatus

logger = get_logger(__name__)

ProgressCallback = Callable[[DownloadProgress], None]


@dataclass
class ModelManagementService:
    """Coordinates discovery, download, validation, and cache for production models."""

    registry: CentralModelRegistry
    models_dir: Path
    enricher: RegistryEnricher
    downloader: DownloadManager
    cache: ModelCacheService
    validator: InstallValidator
    hardware: HardwareAssessment

    @classmethod
    def create(cls, registry: CentralModelRegistry, models_dir: Path) -> ModelManagementService:
        enricher = RegistryEnricher(models_dir)
        downloader = DownloadManager(models_dir)
        cache = ModelCacheService(models_dir)
        validator = InstallValidator()
        hardware = assess_models()
        service = cls(
            registry=registry,
            models_dir=models_dir,
            enricher=enricher,
            downloader=downloader,
            cache=cache,
            validator=validator,
            hardware=hardware,
        )
        service.refresh()
        return service

    @property
    def records(self) -> tuple[ModelRecord, ...]:
        return self.enricher.refresh_all(self.registry.records)

    def refresh(self) -> None:
        self.registry.refresh()
        self.enricher.refresh_all(self.registry.records)

    def missing_mandatory(self) -> tuple[ModelRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.mandatory
            and record.installation_status == InstallationStatus.NOT_INSTALLED
        )

    def all_mandatory_ready(self) -> bool:
        if not is_online() and self.missing_mandatory():
            installed = tuple(
                record
                for record in self.records
                if record.mandatory and record.installation_status != InstallationStatus.NOT_INSTALLED
            )
            return len(installed) == len([spec for spec in PRODUCTION_MODELS if spec.mandatory])
        return all_mandatory_validated(self.records)

    def offline_status(self) -> OfflineReport:
        return offline_report(self.records)

    def estimated_download_bytes(self) -> int:
        missing = self.missing_mandatory()
        total = 0
        for record in missing:
            spec = spec_for_kind(record.kind)
            total += spec.expected_size_bytes or 0
        return total or total_expected_download_bytes()

    def free_disk_bytes(self) -> int:
        return available_disk_bytes(self.models_dir)

    def download_all(self, *, on_progress: ProgressCallback | None = None) -> None:
        kinds = tuple(record.kind for record in self.missing_mandatory())
        if kinds:
            self.downloader.download_models(kinds, on_progress=on_progress)

    def download_selected(
        self,
        kinds: tuple[ModelKind, ...],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.downloader.download_models(kinds, on_progress=on_progress)

    def wait_for_downloads(self, timeout_seconds: float | None = None) -> bool:
        finished = self.downloader.wait_for_completion(timeout_seconds)
        if finished:
            self.refresh()
            for record in self.records:
                result = self.validator.validate_record(record, self.models_dir)
                if not result.passed:
                    logger.warning("Validation failed for %s: %s", record.display_name, result.detail)
        return finished

    def validate_installed(self) -> tuple[ModelRecord, ...]:
        validated: list[ModelRecord] = []
        for record in self.records:
            result = self.validator.validate_record(record, self.models_dir)
            validated.append(result.record)
        return tuple(validated)

    def repair_model(self, kind: ModelKind) -> None:
        self.cache.repair(kind, self.downloader)
        self.refresh()

    def uninstall_model(self, kind: ModelKind) -> None:
        self.cache.uninstall(kind)
        self.refresh()
