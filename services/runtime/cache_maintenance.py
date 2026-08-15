"""Cache maintenance and cleanup utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config.app_config import AppConfig
from core.logging import get_logger
from services.cache.cache_manager import CacheManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheSizeReport:
    """Size summary for cache and temporary storage."""

    cache_bytes: int
    cache_files: int
    temp_bytes: int
    temp_files: int
    orphaned_files: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return self.cache_bytes + self.temp_bytes


@dataclass(frozen=True)
class CacheCleanupReport:
    """Outcome of a safe cache cleanup operation."""

    removed_cache_files: int
    removed_temp_files: int
    removed_orphans: int
    freed_bytes: int


class CacheMaintenanceService:
    """Reports cache usage and performs safe cleanup."""

    _KNOWN_CACHE_SUFFIXES = {".json"}

    def __init__(self, app_config: AppConfig, cache_manager: CacheManager) -> None:
        self._cache_dir = app_config.paths.cache_dir
        self._temp_dir = app_config.paths.cache_dir.parent / "tmp"
        self._cache_manager = cache_manager

    def report_size(self) -> CacheSizeReport:
        cache_bytes, cache_files = self._directory_size(self._cache_dir)
        temp_bytes, temp_files = self._directory_size(self._temp_dir)
        return CacheSizeReport(
            cache_bytes=cache_bytes,
            cache_files=cache_files,
            temp_bytes=temp_bytes,
            temp_files=temp_files,
            orphaned_files=self.detect_orphans(),
        )

    def detect_orphans(self) -> tuple[str, ...]:
        orphans: list[str] = []
        if not self._cache_dir.exists():
            return ()
        for path in self._cache_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._KNOWN_CACHE_SUFFIXES:
                orphans.append(path.name)
        return tuple(sorted(orphans))

    def safe_cleanup(self) -> CacheCleanupReport:
        """Remove corrupt/orphan cache entries and stale temporary files."""
        removed_cache = 0
        removed_temp = 0
        removed_orphans = 0
        freed_bytes = 0

        for orphan_name in self.detect_orphans():
            path = self._cache_dir / orphan_name
            try:
                freed_bytes += path.stat().st_size
                path.unlink(missing_ok=True)
                removed_orphans += 1
            except OSError:
                logger.warning("Could not remove orphaned cache file: %s", path)

        if self._cache_dir.exists():
            for path in self._cache_dir.glob("*.json"):
                if self._cache_manager.read_json(path.stem) is None and path.exists():
                    try:
                        freed_bytes += path.stat().st_size
                        path.unlink(missing_ok=True)
                        removed_cache += 1
                    except OSError:
                        logger.warning("Could not remove invalid cache file: %s", path)

        if self._temp_dir.exists():
            for path in self._temp_dir.glob("*"):
                if not path.is_file():
                    continue
                try:
                    freed_bytes += path.stat().st_size
                    path.unlink(missing_ok=True)
                    removed_temp += 1
                except OSError:
                    logger.warning("Could not remove temporary file: %s", path)

        logger.info(
            "Cache cleanup removed %d cache, %d temp, %d orphan files",
            removed_cache,
            removed_temp,
            removed_orphans,
        )
        return CacheCleanupReport(
            removed_cache_files=removed_cache,
            removed_temp_files=removed_temp,
            removed_orphans=removed_orphans,
            freed_bytes=freed_bytes,
        )

    @staticmethod
    def _directory_size(directory: Path) -> tuple[int, int]:
        if not directory.exists():
            return 0, 0
        total = 0
        count = 0
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                    count += 1
                except OSError:
                    continue
        return total, count
