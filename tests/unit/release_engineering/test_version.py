"""Unit tests for centralized version constants."""

from release.version import (
    AI_PIPELINE_VERSION,
    APPLICATION_VERSION,
    ARCHITECTURE_VERSION,
    CONFIGURATION_VERSION,
    MODEL_REGISTRY_VERSION,
)


def test_version_constants_are_defined() -> None:
    assert APPLICATION_VERSION == "1.0.0"
    assert ARCHITECTURE_VERSION == "2.3"
    assert AI_PIPELINE_VERSION == "1.0.0"
    assert MODEL_REGISTRY_VERSION == "1.0.0"
    assert CONFIGURATION_VERSION == "1.0.0"
