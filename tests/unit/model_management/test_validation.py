"""Tests for install validation."""

from pathlib import Path

import pytest

from core.constants.model_kinds import ModelKind
from model_management.validation import InstallValidator
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus, ModelRuntimeStatus


def _yolo_record() -> ModelRecord:
    return ModelRecord(
        kind=ModelKind.YOLO,
        identifier="yolo11x",
        display_name="Ultralytics YOLO11x",
        version="11.x",
        provider="Ultralytics",
        supported_tasks=("YOLO_DETECTION",),
        file_location=None,
        device_compatibility=("cuda", "cpu"),
        runtime_status=ModelRuntimeStatus.MISSING,
        installation_status=InstallationStatus.NOT_INSTALLED,
        mandatory=True,
    )


def test_corrupted_empty_yolo_removed(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bad = models_dir / "yolo11x.pt"
    bad.write_bytes(b"")
    validator = InstallValidator()
    result = validator.validate_record(_yolo_record(), models_dir)
    assert not result.passed
    assert not bad.exists()


def test_valid_yolo_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from model_management.catalog import spec_for_kind

    spec = spec_for_kind(ModelKind.YOLO)
    monkeypatch.setattr(
        "model_management.validation.spec_for_kind",
        lambda kind: replace(spec, expected_size_bytes=2048),
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    good = models_dir / "yolo11x.pt"
    good.write_bytes(b"\x00" * 2048)
    validator = InstallValidator()
    result = validator.validate_record(_yolo_record(), models_dir)
    assert result.passed
    assert result.record.file_location == good
