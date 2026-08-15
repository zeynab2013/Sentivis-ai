"""Memory monitoring and cleanup."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import psutil

from core.config.app_config import AppConfig
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory statistics."""

    ram_used_mb: float
    ram_available_mb: float
    vram_allocated_mb: float
    vram_reserved_mb: float


class MemoryManager:
    """Tracks memory usage and performs cleanup operations."""

    def __init__(self, config: AppConfig) -> None:
        self._vram_warning = config.hardware.vram_warning_ratio
        self._ram_warning = config.hardware.ram_warning_ratio
        self._release_threshold_mb = config.competition.vram_release_threshold_mb
        self._peak_vram = 0.0
        self._peak_ram = 0.0

    @property
    def peak_ram_mb(self) -> float:
        return self._peak_ram

    @property
    def peak_vram_mb(self) -> float:
        return self._peak_vram

    def reset_peaks(self) -> None:
        self._peak_ram = 0.0
        self._peak_vram = 0.0

    def snapshot(self) -> MemorySnapshot:
        """Capture current RAM and VRAM usage."""
        process = psutil.Process()
        mem = process.memory_info()
        ram_used_mb = mem.rss / (1024 * 1024)
        ram_available_mb = psutil.virtual_memory().available / (1024 * 1024)
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        except ImportError:
            pass

        self._peak_ram = max(self._peak_ram, ram_used_mb)
        self._peak_vram = max(self._peak_vram, vram_allocated_mb)
        return MemorySnapshot(
            ram_used_mb=ram_used_mb,
            ram_available_mb=ram_available_mb,
            vram_allocated_mb=vram_allocated_mb,
            vram_reserved_mb=vram_reserved_mb,
        )

    def clear_gpu_cache(self) -> None:
        """Run garbage collection and clear CUDA cache."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.debug("GPU cache cleared")

    def verify_gpu_released(self) -> bool:
        """Return True when allocated VRAM is below the release threshold."""
        snap = self.snapshot()
        released = snap.vram_allocated_mb <= self._release_threshold_mb
        if not released:
            logger.warning(
                "GPU memory may not be fully released: %.1f MB allocated (threshold %.1f MB)",
                snap.vram_allocated_mb,
                self._release_threshold_mb,
            )
        return released

    def log_snapshot(self, label: str) -> MemorySnapshot:
        """Log and return a memory snapshot."""
        snap = self.snapshot()
        logger.info(
            "%s | RAM %.1f MB | VRAM alloc %.1f MB reserved %.1f MB",
            label,
            snap.ram_used_mb,
            snap.vram_allocated_mb,
            snap.vram_reserved_mb,
        )
        self._check_warnings(snap)
        return snap

    def log_peak(self, run_id: str) -> None:
        """Log peak memory for a pipeline run."""
        logger.info(
            "Peak memory run=%s RAM=%.1f MB VRAM=%.1f MB",
            run_id,
            self._peak_ram,
            self._peak_vram,
        )

    def recover_from_oom(self) -> None:
        """Force cleanup after out-of-memory conditions."""
        logger.warning("Performing OOM recovery cleanup")
        self.clear_gpu_cache()

    def _check_warnings(self, snap: MemorySnapshot) -> None:
        total_ram = snap.ram_used_mb + snap.ram_available_mb
        if total_ram > 0 and (snap.ram_used_mb / total_ram) >= self._ram_warning:
            logger.warning("RAM usage above threshold")
        if snap.vram_reserved_mb > 0 and snap.vram_allocated_mb / max(snap.vram_reserved_mb, 1) >= self._vram_warning:
            logger.warning("VRAM usage above threshold")
