"""Tests for download manager behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.constants.model_kinds import ModelKind
from model_management.catalog import spec_for_kind
from model_management.download.manager import DownloadManager
from model_management.download.progress import DownloadState


@pytest.fixture
def manager(tmp_path: Path) -> DownloadManager:
    return DownloadManager(tmp_path / "models")


def test_missing_yolo_not_installed(manager: DownloadManager) -> None:
    spec = spec_for_kind(ModelKind.YOLO)
    assert not manager.is_model_installed(spec)


def test_cancel_download(manager: DownloadManager) -> None:
    progress_states: list[DownloadState] = []

    def on_progress(progress: object) -> None:
        from model_management.download.progress import DownloadProgress

        if isinstance(progress, DownloadProgress):
            progress_states.append(progress.state)
            manager.cancel()

    with patch("model_management.download.manager.download_yolo_weights", side_effect=OSError("network")):
        manager.download_models((ModelKind.YOLO,), on_progress=on_progress)
        manager.wait_for_completion(timeout_seconds=5)
    assert DownloadState.FAILED in progress_states or DownloadState.CANCELLED in progress_states


def test_yolo_marked_installed_when_file_exists(manager: DownloadManager, tmp_path: Path) -> None:
    spec = spec_for_kind(ModelKind.YOLO)
    path = manager.models_dir / spec.local_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 1024)
    assert manager.is_model_installed(spec)


def test_available_disk_bytes(manager: DownloadManager) -> None:
    from model_management.download.manager import available_disk_bytes

    assert available_disk_bytes(manager.models_dir) > 0
