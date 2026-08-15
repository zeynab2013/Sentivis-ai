"""Background model download orchestration."""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from core.constants.model_kinds import ModelKind
from core.logging import get_logger
from model_management.catalog import DownloadSource, ProductionModelSpec, spec_for_kind
from model_management.download.progress import DownloadProgress, DownloadState
from model_management.download.sources.huggingface import download_hf_model, is_hf_model_cached
from model_management.download.sources.ollama import pull_ollama_model
from model_management.download.sources.ultralytics import download_yolo_weights

logger = get_logger(__name__)

ProgressCallback = Callable[[DownloadProgress], None]


class DownloadManager:
    """Manages resumable background downloads for production models."""

    def __init__(self, models_dir: Path, *, max_retries: int = 3) -> None:
        self._models_dir = models_dir
        self._max_retries = max_retries
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_progress: dict[str, DownloadProgress] = {}

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def progress_for(self, display_name: str) -> DownloadProgress | None:
        return self._last_progress.get(display_name)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel_event.set()

    def download_models(
        self,
        kinds: tuple[ModelKind, ...],
        *,
        on_progress: ProgressCallback | None = None,
        gemma_via_ollama: bool = False,
    ) -> None:
        """Start background download for selected model kinds."""
        if self.is_running():
            return
        self._cancel_event.clear()

        def worker() -> None:
            for kind in kinds:
                if self._cancel_event.is_set():
                    break
                spec = spec_for_kind(kind)
                self._download_with_retry(spec, on_progress=on_progress, gemma_via_ollama=gemma_via_ollama)

        self._thread = threading.Thread(target=worker, name="ModelDownloadWorker", daemon=True)
        self._thread.start()

    def wait_for_completion(self, timeout_seconds: float | None = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout_seconds)
        return not self._thread.is_alive()

    def is_model_installed(self, spec: ProductionModelSpec) -> bool:
        if spec.download_source == DownloadSource.ULTRALYTICS:
            path = self._models_dir / spec.local_filename
            return path.is_file() and path.stat().st_size > 0
        if spec.download_source == DownloadSource.HUGGINGFACE:
            return is_hf_model_cached(spec.hf_repo_id or spec.model_id)
        if spec.download_source == DownloadSource.OLLAMA:
            return False
        return False

    def _download_with_retry(
        self,
        spec: ProductionModelSpec,
        *,
        on_progress: ProgressCallback | None,
        gemma_via_ollama: bool,
    ) -> None:
        for attempt in range(1, self._max_retries + 1):
            if self._cancel_event.is_set():
                self._emit(
                    DownloadProgress(spec.display_name, DownloadState.CANCELLED, attempt=attempt),
                    on_progress,
                )
                return
            try:
                self._download_one(spec, on_progress=on_progress, gemma_via_ollama=gemma_via_ollama)
                return
            except (OSError, PermissionError) as exc:
                logger.warning(
                    "Download attempt %s/%s failed for %s: %s",
                    attempt,
                    self._max_retries,
                    spec.display_name,
                    exc,
                )
                self._emit(
                    DownloadProgress(
                        spec.display_name,
                        DownloadState.FAILED,
                        message=str(exc),
                        attempt=attempt,
                    ),
                    on_progress,
                )
                if attempt < self._max_retries:
                    time.sleep(min(2**attempt, 10))
        self._cleanup_failed(spec)

    def _download_one(
        self,
        spec: ProductionModelSpec,
        *,
        on_progress: ProgressCallback | None,
        gemma_via_ollama: bool,
    ) -> None:
        self._emit(
            DownloadProgress(spec.display_name, DownloadState.RUNNING, message="Starting download…"),
            on_progress,
        )
        if spec.kind == ModelKind.GEMMA and gemma_via_ollama:
            pull_ollama_model(spec, on_progress=on_progress)
            return
        if spec.download_source == DownloadSource.ULTRALYTICS:
            download_yolo_weights(spec, self._models_dir, on_progress=on_progress)
            return
        if spec.download_source == DownloadSource.HUGGINGFACE:
            download_hf_model(spec, on_progress=on_progress)
            return
        raise OSError(f"Unsupported download source for {spec.display_name}")

    def _cleanup_failed(self, spec: ProductionModelSpec) -> None:
        if spec.download_source != DownloadSource.ULTRALYTICS:
            return
        path = self._models_dir / spec.local_filename
        partial = path.with_suffix(path.suffix + ".partial")
        for candidate in (path, partial):
            if candidate.is_file() and candidate.stat().st_size == 0:
                candidate.unlink(missing_ok=True)

    def _emit(self, progress: DownloadProgress, callback: ProgressCallback | None) -> None:
        with self._lock:
            self._last_progress[progress.model_name] = progress
        if callback is not None:
            callback(progress)


def available_disk_bytes(path: Path) -> int:
    """Return free disk space for download target."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free
