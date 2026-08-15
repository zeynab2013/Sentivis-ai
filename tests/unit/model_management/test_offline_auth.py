"""Tests for offline mode and auth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.constants.model_kinds import ModelKind
from model_management.auth import clear_hf_token, store_hf_token, token_from_secure_store
from model_management.offline import is_online, offline_report
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus, ModelRuntimeStatus


def _record(name: str, status: InstallationStatus) -> ModelRecord:
    return ModelRecord(
        kind=ModelKind.YOLO,
        identifier="yolo11x",
        display_name=name,
        version="11.x",
        provider="Ultralytics",
        supported_tasks=(),
        file_location=None,
        device_compatibility=("cpu",),
        runtime_status=ModelRuntimeStatus.MISSING,
        installation_status=status,
        mandatory=True,
    )


def test_offline_report_lists_missing(tmp_path: Path) -> None:
    records = (_record("YOLO11x", InstallationStatus.NOT_INSTALLED),)
    with patch("model_management.offline.is_online", return_value=False):
        report = offline_report(records)
    assert report.offline is True
    assert "YOLO11x" in report.missing_models[0]


def test_hf_token_store_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")
    from core.config import user_config_paths

    monkeypatch.setattr(user_config_paths, "user_config_dir", lambda: tmp_path)
    store_hf_token("test-token-value")
    assert token_from_secure_store() == "test-token-value"
    clear_hf_token()
    assert token_from_secure_store() is None


def test_is_online_handles_failure() -> None:
    with patch("model_management.offline.socket.create_connection", side_effect=OSError("offline")):
        assert is_online() is False
