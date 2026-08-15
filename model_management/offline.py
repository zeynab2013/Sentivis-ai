"""Offline mode detection and messaging."""

from __future__ import annotations

import socket
from dataclasses import dataclass

from core.constants.model_kinds import ModelKind
from model_management.catalog import PRODUCTION_MODELS
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus


@dataclass(frozen=True)
class OfflineReport:
    """Models unavailable without network."""

    offline: bool
    missing_models: tuple[str, ...]
    message: str


def is_online(timeout_seconds: float = 2.0) -> bool:
    """Best-effort internet connectivity probe."""
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def offline_report(records: tuple[ModelRecord, ...]) -> OfflineReport:
    """Build user-facing offline availability report."""
    missing = tuple(
        record.display_name
        for record in records
        if record.installation_status != InstallationStatus.VALIDATED
        and record.installation_status != InstallationStatus.INSTALLED
    )
    if is_online():
        return OfflineReport(offline=False, missing_models=missing, message="Network available.")
    if not missing:
        return OfflineReport(
            offline=True,
            missing_models=(),
            message="Offline mode — using installed models.",
        )
    names = ", ".join(missing)
    return OfflineReport(
        offline=True,
        missing_models=missing,
        message=f"Offline — required models unavailable: {names}",
    )


def mandatory_kinds() -> tuple[ModelKind, ...]:
    return tuple(spec.kind for spec in PRODUCTION_MODELS if spec.mandatory)
