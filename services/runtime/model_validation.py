"""Detailed model validation before inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from services.models.device_selector import DeviceSelector
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import IntegrityStatus, ModelRuntimeStatus
from services.runtime.yolo_weights import resolve_yolo_weights_path


@dataclass(frozen=True)
class ValidationIssue:
    """One precise validation failure."""

    code: str
    message: str


@dataclass(frozen=True)
class ModelValidationResult:
    """Outcome of validating one model record."""

    record: ModelRecord
    passed: bool
    issues: tuple[ValidationIssue, ...]
    checksum: str | None = None

    @property
    def failure_summary(self) -> str:
        if self.passed:
            return "Validation passed"
        return "; ".join(issue.message for issue in self.issues)


class ModelValidationService:
    """Validates model availability, files, configuration, and integrity."""

    _SUPPORTED_PLUGIN_VERSIONS = ("1.0.0",)

    def __init__(self, model_config: ModelConfig, device_selector: DeviceSelector) -> None:
        self._config = model_config
        self._device_selector = device_selector

    def validate(self, record: ModelRecord, *, plugin_version: str | None = None) -> ModelValidationResult:
        issues: list[ValidationIssue] = []
        checksum: str | None = None
        now = datetime.now(tz=timezone.utc)  # noqa: UP017

        if record.kind == ModelKind.YOLO:
            yolo_issues, resolved_weights = self._validate_yolo(record)
            issues.extend(yolo_issues)
            if resolved_weights is not None and resolved_weights.is_file():
                checksum = self._checksum(resolved_weights)
                record = record.with_status(record.runtime_status, file_location=resolved_weights)
        elif record.kind == ModelKind.BLIP:
            issues.extend(self._validate_remote_model(record, self._config.blip.model_id))
        elif record.kind == ModelKind.GEMMA:
            issues.extend(self._validate_remote_model(record, self._config.gemma.model_id))

        if plugin_version is not None and plugin_version not in self._SUPPORTED_PLUGIN_VERSIONS:
            issues.append(
                ValidationIssue(
                    code="unsupported_version",
                    message=f"Plugin version {plugin_version} is not supported",
                )
            )

        preferred = self._preferred_device(record.kind)
        device = self._device_selector.preferred_device(preferred)
        if preferred == "cuda" and device != "cuda":
            issues.append(
                ValidationIssue(
                    code="device_unavailable",
                    message="CUDA requested but unavailable; CPU fallback will be used",
                )
            )

        blocking = tuple(
            issue for issue in issues if issue.code not in {"device_unavailable", "missing_file"}
        )
        passed = not blocking
        integrity = IntegrityStatus.VALID if passed else IntegrityStatus.INVALID
        if record.file_location is None and record.kind != ModelKind.YOLO:
            integrity = IntegrityStatus.SKIPPED if passed else IntegrityStatus.INVALID

        updated = record.with_status(
            ModelRuntimeStatus.READY if passed else ModelRuntimeStatus.VALIDATION_FAILED,
            integrity_status=integrity,
            validation_detail="; ".join(issue.message for issue in issues) or "Validation passed",
            last_validation_time=now,
        )
        return ModelValidationResult(record=updated, passed=passed, issues=tuple(issues), checksum=checksum)

    def _preferred_device(self, kind: ModelKind) -> str:
        if kind == ModelKind.YOLO:
            return self._config.yolo.preferred_device
        if kind == ModelKind.BLIP:
            return self._config.blip.preferred_device
        return self._config.gemma.preferred_device

    def _validate_yolo(self, record: ModelRecord) -> tuple[list[ValidationIssue], Path | None]:
        issues: list[ValidationIssue] = []
        if not self._config.yolo.variant.strip():
            issues.append(ValidationIssue("missing_config", "YOLO variant is not configured"))

        resolved = resolve_yolo_weights_path(
            variant=self._config.yolo.variant,
            configured_path=self._config.yolo.weights_path,
            search_paths=record.search_paths,
        )
        if resolved is None:
            issues.append(
                ValidationIssue(
                    "missing_file",
                    "No local YOLO weights found; first run may download weights automatically",
                )
            )
            return issues, None

        if not resolved.is_file():
            issues.append(
                ValidationIssue("unreadable_file", f"YOLO weights path is not a file: {resolved}")
            )
            return issues, resolved

        try:
            resolved.open("rb").read(1)
        except OSError as exc:
            issues.append(
                ValidationIssue("unreadable_file", f"YOLO weights are not readable: {exc}")
            )
        return issues, resolved

    def _validate_remote_model(
        self,
        record: ModelRecord,
        model_id: str,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not model_id.strip():
            issues.append(
                ValidationIssue("missing_config", f"Model ID is not configured for {record.display_name}")
            )
        elif record.identifier != model_id:
            issues.append(
                ValidationIssue(
                    "config_mismatch",
                    f"Registry identifier {record.identifier!r} does not match configuration {model_id!r}",
                )
            )
        return issues

    @staticmethod
    def _checksum(path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None
