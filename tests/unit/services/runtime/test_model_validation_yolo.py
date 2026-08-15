"""Tests for YOLO validation with discovered weights."""

from __future__ import annotations

from pathlib import Path

from core.config.loader import load_app_config, load_model_config
from core.constants.model_kinds import ModelKind
from services.models.device_selector import DeviceSelector
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import IntegrityStatus, ModelRuntimeStatus
from services.runtime.model_validation import ModelValidationService


def test_yolo_validation_passes_when_weights_discovered(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weights = models_dir / "yolo11x.pt"
    weights.write_bytes(b"x" * 4096)

    model_config = load_model_config()
    validator = ModelValidationService(model_config, DeviceSelector(load_app_config()))
    record = ModelRecord(
        kind=ModelKind.YOLO,
        identifier="yolo11x",
        display_name="Ultralytics YOLO11x",
        version="11.x",
        provider="Ultralytics",
        supported_tasks=("YOLO_DETECTION",),
        file_location=None,
        device_compatibility=("cuda", "cpu"),
        runtime_status=ModelRuntimeStatus.INSTALLED,
        integrity_status=IntegrityStatus.UNKNOWN,
        search_paths=(models_dir.resolve(),),
    )

    result = validator.validate(record, plugin_version="1.0.0")
    assert result.passed
    assert result.record.file_location == weights.resolve()
    assert "YOLO weights path is not a file: ." not in result.failure_summary
