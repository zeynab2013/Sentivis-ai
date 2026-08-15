"""Central model registry record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.constants.model_kinds import ModelKind
from services.runtime.model_status import InstallationStatus, IntegrityStatus, ModelRuntimeStatus


@dataclass
class ModelRecord:
    """Metadata tracked for one configured AI model."""

    kind: ModelKind
    identifier: str
    display_name: str
    version: str
    provider: str
    supported_tasks: tuple[str, ...]
    file_location: Path | None
    device_compatibility: tuple[str, ...]
    runtime_status: ModelRuntimeStatus = ModelRuntimeStatus.MISSING
    integrity_status: IntegrityStatus = IntegrityStatus.UNKNOWN
    last_validation_time: datetime | None = None
    validation_detail: str = ""
    search_paths: tuple[Path, ...] = field(default_factory=tuple)
    download_source: str = ""
    expected_size_bytes: int | None = None
    checksum: str | None = None
    license_name: str = ""
    installation_status: InstallationStatus = InstallationStatus.NOT_INSTALLED
    quantization: str | None = None
    mandatory: bool = True

    def with_status(
        self,
        runtime_status: ModelRuntimeStatus,
        *,
        integrity_status: IntegrityStatus | None = None,
        validation_detail: str | None = None,
        last_validation_time: datetime | None = None,
        installation_status: InstallationStatus | None = None,
        checksum: str | None = None,
        file_location: Path | None = None,
    ) -> ModelRecord:
        return ModelRecord(
            kind=self.kind,
            identifier=self.identifier,
            display_name=self.display_name,
            version=self.version,
            provider=self.provider,
            supported_tasks=self.supported_tasks,
            file_location=file_location if file_location is not None else self.file_location,
            device_compatibility=self.device_compatibility,
            runtime_status=runtime_status,
            integrity_status=integrity_status or self.integrity_status,
            last_validation_time=last_validation_time or self.last_validation_time,
            validation_detail=validation_detail or self.validation_detail,
            search_paths=self.search_paths,
            download_source=self.download_source,
            expected_size_bytes=self.expected_size_bytes,
            checksum=checksum if checksum is not None else self.checksum,
            license_name=self.license_name,
            installation_status=installation_status or self.installation_status,
            quantization=self.quantization,
            mandatory=self.mandatory,
        )
