"""Ultralytics YOLO weight downloader."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from core.logging import get_logger
from model_management.catalog import ProductionModelSpec
from model_management.download.progress import DownloadProgress, DownloadState

logger = get_logger(__name__)


def download_yolo_weights(
    spec: ProductionModelSpec,
    models_dir: Path,
    *,
    on_progress: Callable[[DownloadProgress], None] | None = None,
) -> Path:
    """Download YOLO weights via official Ultralytics channel."""
    destination = models_dir / spec.local_filename
    models_dir.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    if on_progress:
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.RUNNING,
                message=f"Downloading {spec.local_filename} via Ultralytics…",
            )
        )

    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(spec.local_filename)
    source = Path(getattr(model, "ckpt_path", None) or spec.local_filename)
    if not source.is_file():
        source = Path(spec.local_filename)
    if source.is_file() and source.resolve() != destination.resolve() or not destination.is_file() and source.is_file():
        shutil.copy2(source, destination)

    if not destination.is_file():
        raise OSError(f"YOLO weights not available after download: {destination}")

    logger.info("YOLO weights installed at %s", destination)
    if on_progress:
        size = destination.stat().st_size
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.COMPLETED,
                bytes_downloaded=size,
                total_bytes=size,
                message="YOLO download complete",
            )
        )
    return destination
