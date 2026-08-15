"""Registry enrichment from production catalog."""

from __future__ import annotations

from pathlib import Path

from core.constants.model_kinds import ModelKind
from model_management.catalog import ProductionModelSpec, spec_for_kind
from model_management.download.manager import DownloadManager
from model_management.download.sources.huggingface import is_hf_model_cached
from model_management.validation import InstallValidator
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus, ModelRuntimeStatus
from services.runtime.yolo_weights import resolve_yolo_weights_path


class RegistryEnricher:
    """Applies production catalog metadata and installation state to registry records."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._validator = InstallValidator()
        self._downloader = DownloadManager(models_dir)

    def enrich(self, record: ModelRecord) -> ModelRecord:
        spec = spec_for_kind(record.kind)
        file_location = record.file_location
        if record.kind == ModelKind.YOLO:
            resolved = resolve_yolo_weights_path(
                variant=spec.model_id,
                configured_path=record.file_location,
                search_paths=(self._models_dir.resolve(), *record.search_paths),
            )
            if resolved is not None:
                file_location = resolved
        enriched = ModelRecord(
            kind=record.kind,
            identifier=spec.model_id,
            display_name=spec.display_name,
            version=spec.version,
            provider=spec.provider,
            supported_tasks=record.supported_tasks,
            file_location=file_location,
            device_compatibility=record.device_compatibility,
            runtime_status=record.runtime_status,
            integrity_status=record.integrity_status,
            last_validation_time=record.last_validation_time,
            validation_detail=record.validation_detail,
            search_paths=record.search_paths,
            download_source=spec.download_source.value,
            expected_size_bytes=spec.expected_size_bytes,
            checksum=record.checksum,
            license_name=spec.license_name,
            installation_status=self._installation_status(spec),
            quantization=spec.quantization,
            mandatory=spec.mandatory,
        )
        if enriched.installation_status == InstallationStatus.INSTALLED:
            result = self._validator.validate_record(enriched, self._models_dir)
            return result.record
        return enriched

    def refresh_all(self, records: tuple[ModelRecord, ...]) -> tuple[ModelRecord, ...]:
        return tuple(self.enrich(record) for record in records)

    def _installation_status(self, spec: ProductionModelSpec) -> InstallationStatus:
        if self._downloader.is_model_installed(spec):
            return InstallationStatus.INSTALLED
        if spec.download_source.value == "huggingface" and is_hf_model_cached(spec.hf_repo_id or spec.model_id):
            return InstallationStatus.INSTALLED
        yolo_path = self._models_dir / spec.local_filename
        if yolo_path.is_file() and yolo_path.stat().st_size > 0:
            return InstallationStatus.INSTALLED
        return InstallationStatus.NOT_INSTALLED


def all_mandatory_validated(records: tuple[ModelRecord, ...]) -> bool:
    """Return True when every mandatory model is validated."""
    for record in records:
        if not record.mandatory:
            continue
        if record.installation_status not in {
            InstallationStatus.VALIDATED,
            InstallationStatus.INSTALLED,
        }:
            return False
        if record.runtime_status not in {ModelRuntimeStatus.READY, ModelRuntimeStatus.INSTALLED}:
            if (
                record.kind in {ModelKind.BLIP, ModelKind.GEMMA}
                and record.installation_status == InstallationStatus.INSTALLED
            ):
                continue
            return False
    return True
