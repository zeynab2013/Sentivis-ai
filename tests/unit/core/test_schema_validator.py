"""Configuration schema validation tests."""

from typing import cast

from core.config.schema_validator import validate_app_config, validate_model_config
from core.exceptions.config import ConfigurationError


def _minimal_app_config() -> dict[str, object]:
    return {
        "app": {"name": "Test", "version": "1.0"},
        "logging": {"level": "INFO"},
        "image": {
            "max_dimension": 1024,
            "max_file_size_bytes": 1048576,
            "yolo_inference_size": 640,
        },
        "hardware": {
            "vram_warning_ratio": 0.85,
            "ram_warning_ratio": 0.9,
            "cpu_fallback_enabled": True,
            "pipeline_timeout_seconds": 600,
        },
        "paths": {},
        "workers": {},
        "competition": {
            "quality_threshold": 0.55,
            "max_hallucination_risk": 0.25,
            "deterministic_seed": 42,
            "gemma_temperature": 0.0,
            "vram_release_threshold_mb": 64.0,
        },
    }


def test_validate_app_config_accepts_minimal_structure() -> None:
    validate_app_config(_minimal_app_config())


def test_validate_app_config_rejects_invalid_vram_ratio() -> None:
    payload = _minimal_app_config()
    hardware = dict(cast(dict[str, object], payload["hardware"]))
    hardware["vram_warning_ratio"] = 1.5
    payload["hardware"] = hardware
    try:
        validate_app_config(payload)
    except ConfigurationError:
        return
    raise AssertionError("Expected ConfigurationError")


def test_validate_model_config_rejects_invalid_confidence() -> None:
    payload = {
        "yolo": {
            "variant": "yolov8n.pt",
            "confidence_threshold": 1.5,
            "iou_threshold": 0.5,
            "preferred_device": "cuda",
        },
        "blip": {"model_id": "x", "preferred_device": "cuda", "max_length": 32},
        "gemma": {
            "model_id": "y",
            "preferred_device": "cuda",
            "quantization": "int4",
            "max_new_tokens": 32,
            "temperature": 0.1,
        },
    }
    try:
        validate_model_config(cast(dict[str, object], payload))
    except ConfigurationError:
        return
    raise AssertionError("Expected ConfigurationError")
