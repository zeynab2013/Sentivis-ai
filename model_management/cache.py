"""Model cache management operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.constants.model_kinds import ModelKind
from core.logging import get_logger
from model_management.catalog import spec_for_kind
from model_management.download.manager import DownloadManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheReport:
    """Cache and model storage summary."""

    models_dir: Path
    models_bytes: int
    partial_bytes: int
    model_files: tuple[str, ...]


class ModelCacheService:
    """Cache inspection and maintenance for managed models."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def report(self) -> CacheReport:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        model_bytes = 0
        partial_bytes = 0
        names: list[str] = []
        for path in self._models_dir.iterdir():
            if not path.is_file():
                continue
            size = path.stat().st_size
            if path.name.endswith(".partial"):
                partial_bytes += size
            else:
                model_bytes += size
                names.append(path.name)
        return CacheReport(
            models_dir=self._models_dir,
            models_bytes=model_bytes,
            partial_bytes=partial_bytes,
            model_files=tuple(sorted(names)),
        )

    def cleanup_partials(self) -> int:
        removed = 0
        for path in self._models_dir.glob("*.partial"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def cleanup_orphans(self, allowed_filenames: tuple[str, ...]) -> int:
        removed = 0
        allowed = set(allowed_filenames)
        for path in self._models_dir.glob("*.pt"):
            if path.name not in allowed:
                path.unlink(missing_ok=True)
                removed += 1
                logger.info("Removed orphan weight file %s", path.name)
        return removed

    def uninstall(self, kind: ModelKind) -> None:
        spec = spec_for_kind(kind)
        if spec.local_filename:
            target = self._models_dir / spec.local_filename
            if target.is_file():
                target.unlink()
        partial = self._models_dir / f"{spec.local_filename}.partial"
        if partial.is_file():
            partial.unlink()

    def repair(self, kind: ModelKind, downloader: DownloadManager) -> None:
        self.uninstall(kind)
        downloader.download_models((kind,))
        downloader.wait_for_completion(timeout_seconds=3600)
