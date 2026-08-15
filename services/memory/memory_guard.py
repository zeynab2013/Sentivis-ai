"""Memory availability checks before heavy pipeline stages."""

from core.config.app_config import AppConfig
from core.constants.pipeline_stages import PipelineStage
from core.exceptions.language import InferenceError
from core.logging import get_logger
from services.memory.memory_manager import MemoryManager, MemorySnapshot

logger = get_logger(__name__)

_STAGE_VRAM_ESTIMATE_MB: dict[PipelineStage, float] = {
    PipelineStage.YOLO_DETECTION: 400.0,
    PipelineStage.BLIP_UNDERSTANDING: 1100.0,
    PipelineStage.GEMMA_REASONING: 1100.0,
}

_MIN_RAM_MB = 512.0


class MemoryGuard:
    """Estimates memory requirements and triggers cleanup when needed."""

    def __init__(self, config: AppConfig, memory_manager: MemoryManager) -> None:
        self._vram_warning = config.hardware.vram_warning_ratio
        self._ram_warning = config.hardware.ram_warning_ratio
        self._memory = memory_manager

    def ensure_stage_capacity(self, stage: PipelineStage) -> MemorySnapshot:
        """Verify memory is sufficient for a heavy stage; cleanup and retry once."""
        snapshot = self._memory.log_snapshot(f"memory guard {stage.name}")
        required_vram = _STAGE_VRAM_ESTIMATE_MB.get(stage, 0.0)
        if required_vram <= 0.0:
            return snapshot

        if self._has_capacity(snapshot, required_vram):
            return snapshot

        logger.warning("Insufficient memory before %s; attempting cleanup", stage.name)
        self._memory.clear_gpu_cache()
        snapshot = self._memory.log_snapshot(f"memory guard retry {stage.name}")

        if self._has_capacity(snapshot, required_vram):
            return snapshot

        raise InferenceError(
            "Not enough memory is available to run this analysis stage.",
            f"Stage {stage.name} requires ~{required_vram:.0f} MB VRAM headroom",
            stage=stage,
            recoverable=True,
        )

    def _has_capacity(self, snapshot: MemorySnapshot, required_vram_mb: float) -> bool:
        total_ram = snapshot.ram_used_mb + snapshot.ram_available_mb
        if snapshot.ram_available_mb < _MIN_RAM_MB:
            return False
        if total_ram > 0 and (snapshot.ram_used_mb / total_ram) >= self._ram_warning:
            logger.warning("RAM usage above configured threshold")

        if required_vram_mb <= 0.0:
            return True

        if snapshot.vram_reserved_mb <= 0.0:
            return True

        available_vram = max(0.0, snapshot.vram_reserved_mb - snapshot.vram_allocated_mb)
        if available_vram < required_vram_mb * 0.25:
            used_ratio = snapshot.vram_allocated_mb / max(snapshot.vram_reserved_mb, 1.0)
            if used_ratio >= self._vram_warning:
                return False
        return True
