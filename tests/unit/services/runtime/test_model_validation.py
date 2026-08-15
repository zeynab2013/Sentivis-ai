"""Unit tests for model validation service."""

from core.config.loader import load_model_config
from core.constants.model_kinds import ModelKind
from services.models.device_selector import DeviceSelector
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import IntegrityStatus, ModelRuntimeStatus
from services.runtime.model_validation import ModelValidationService


def test_model_validation_passes_for_configured_remote_models() -> None:
    model_config = load_model_config()
    from core.config.loader import load_app_config

    validator = ModelValidationService(model_config, DeviceSelector(load_app_config()))
    record = ModelRecord(
        kind=ModelKind.BLIP,
        identifier=model_config.blip.model_id,
        display_name="BLIP",
        version="1.0.0",
        provider="Hugging Face",
        supported_tasks=("BLIP_UNDERSTANDING",),
        file_location=None,
        device_compatibility=("cuda", "cpu"),
        runtime_status=ModelRuntimeStatus.INSTALLED,
        integrity_status=IntegrityStatus.SKIPPED,
    )
    result = validator.validate(record, plugin_version="1.0.0")
    assert result.passed
