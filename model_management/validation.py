"""Post-download model validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.constants.model_kinds import ModelKind
from model_management.catalog import ProductionModelSpec, spec_for_kind
from model_management.download.sources.huggingface import is_hf_model_cached
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus, IntegrityStatus, ModelRuntimeStatus
from services.runtime.yolo_weights import resolve_yolo_weights_path


@dataclass(frozen=True)
class InstallValidationResult:
    """Outcome of validating an installed model."""

    record: ModelRecord
    passed: bool
    detail: str


class InstallValidator:
    """Validates downloaded models before marking them ready."""

    def validate_record(self, record: ModelRecord, models_dir: Path) -> InstallValidationResult:
        spec = spec_for_kind(record.kind)
        if record.kind == ModelKind.YOLO:
            return self._validate_yolo(record, spec, models_dir)
        if record.kind in {ModelKind.BLIP, ModelKind.GEMMA}:
            return self._validate_hf(record, spec)
        return InstallValidationResult(record, False, "Unknown model kind")

    def _validate_yolo(
        self,
        record: ModelRecord,
        spec: ProductionModelSpec,
        models_dir: Path,
    ) -> InstallValidationResult:
        path = resolve_yolo_weights_path(
            variant=spec.model_id,
            configured_path=record.file_location,
            search_paths=(models_dir.resolve(), *record.search_paths),
        )
        if path is None:
            updated = record.with_status(
                ModelRuntimeStatus.MISSING,
                integrity_status=IntegrityStatus.INVALID,
                installation_status=InstallationStatus.NOT_INSTALLED,
                validation_detail="YOLO weights not found",
                last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
            )
            return InstallValidationResult(updated, False, "YOLO weights not found")

        size = path.stat().st_size
        if size <= 0:
            path.unlink(missing_ok=True)
            updated = record.with_status(
                ModelRuntimeStatus.VALIDATION_FAILED,
                integrity_status=IntegrityStatus.CORRUPTED,
                installation_status=InstallationStatus.CORRUPTED,
                validation_detail="YOLO weights file is empty",
                last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
            )
            return InstallValidationResult(updated, False, "Corrupted YOLO weights removed")

        if spec.expected_size_bytes and size < spec.expected_size_bytes * 0.5:
            return InstallValidationResult(
                record.with_status(
                    ModelRuntimeStatus.VALIDATION_FAILED,
                    integrity_status=IntegrityStatus.INVALID,
                    installation_status=InstallationStatus.CORRUPTED,
                    validation_detail="YOLO weights size below expected threshold",
                    last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
                ),
                False,
                "YOLO weights appear incomplete",
            )

        checksum = self._checksum(path)
        if spec.checksum_sha256 and checksum != spec.checksum_sha256:
            path.unlink(missing_ok=True)
            updated = record.with_status(
                ModelRuntimeStatus.VALIDATION_FAILED,
                integrity_status=IntegrityStatus.CORRUPTED,
                installation_status=InstallationStatus.CORRUPTED,
                validation_detail="YOLO checksum mismatch",
                last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
                checksum=checksum,
            )
            return InstallValidationResult(updated, False, "Checksum mismatch — file removed")

        updated = record.with_status(
            ModelRuntimeStatus.READY,
            integrity_status=IntegrityStatus.VALID,
            installation_status=InstallationStatus.VALIDATED,
            validation_detail="YOLO weights validated",
            last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
            checksum=checksum,
            file_location=path.resolve(),
        )
        return InstallValidationResult(updated, True, "YOLO validated")

    def _validate_hf(self, record: ModelRecord, spec: ProductionModelSpec) -> InstallValidationResult:
        repo_id = spec.hf_repo_id or spec.model_id
        from model_management.auth import resolve_hf_token

        token = resolve_hf_token()
        if not is_hf_model_cached(repo_id):
            updated = record.with_status(
                ModelRuntimeStatus.MISSING,
                integrity_status=IntegrityStatus.UNKNOWN,
                installation_status=InstallationStatus.NOT_INSTALLED,
                validation_detail=f"{repo_id} not found in Hugging Face cache",
                last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
            )
            return InstallValidationResult(updated, False, "Model not cached")

        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
        except Exception as exc:
            detail = str(exc)
            if "401" in detail or "403" in detail or "gated" in detail.lower():
                updated = record.with_status(
                    ModelRuntimeStatus.VALIDATION_FAILED,
                    integrity_status=IntegrityStatus.INVALID,
                    installation_status=InstallationStatus.NOT_INSTALLED,
                    validation_detail="Hugging Face authentication required for gated model",
                    last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
                )
                return InstallValidationResult(updated, False, "HF token required for Gemma")
            updated = record.with_status(
                ModelRuntimeStatus.VALIDATION_FAILED,
                integrity_status=IntegrityStatus.INVALID,
                installation_status=InstallationStatus.CORRUPTED,
                validation_detail=f"HF validation failed: {detail}",
                last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
            )
            return InstallValidationResult(updated, False, detail)

        updated = record.with_status(
            ModelRuntimeStatus.READY,
            integrity_status=IntegrityStatus.VALID,
            installation_status=InstallationStatus.VALIDATED,
            validation_detail=f"{repo_id} available in Hugging Face cache",
            last_validation_time=datetime.now(tz=timezone.utc),  # noqa: UP017
        )
        return InstallValidationResult(updated, True, "Hugging Face model validated")

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
