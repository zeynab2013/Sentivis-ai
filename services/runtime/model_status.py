"""Model runtime status enumerations."""

from __future__ import annotations

from enum import Enum


class ModelRuntimeStatus(str, Enum):  # noqa: UP042
    """Lifecycle status exposed to callers and UI adapters."""

    INSTALLED = "installed"
    MISSING = "missing"
    LOADING = "loading"
    READY = "ready"
    IN_USE = "in_use"
    RELEASED = "released"
    UNAVAILABLE = "unavailable"
    VALIDATION_FAILED = "validation_failed"


class IntegrityStatus(str, Enum):  # noqa: UP042
    """Integrity check outcome for model files."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"
    CORRUPTED = "corrupted"


class InstallationStatus(str, Enum):  # noqa: UP042
    """Installation state for managed model assets."""

    NOT_INSTALLED = "not_installed"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    CORRUPTED = "corrupted"
    OUTDATED = "outdated"
    VALIDATED = "validated"
