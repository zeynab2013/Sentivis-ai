"""Application configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path

from core.config.enhancement_config import EnhancementConfig


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behavior settings."""

    level: str
    console_enabled: bool
    max_file_bytes: int
    backup_count: int


@dataclass(frozen=True)
class ImageConfig:
    """Image validation and preprocessing limits."""

    max_dimension: int
    max_file_size_bytes: int
    yolo_inference_size: int
    enhancement: EnhancementConfig

@dataclass(frozen=True)
class HardwareConfig:
    """Hardware resource thresholds."""

    vram_warning_ratio: float
    ram_warning_ratio: float
    cpu_fallback_enabled: bool
    pipeline_timeout_seconds: int


@dataclass(frozen=True)
class PathsConfig:
    """Runtime directory paths."""

    cache_dir: Path
    exports_dir: Path
    logs_dir: Path
    models_dir: Path
    model_search_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CompetitionConfig:
    """Competition and benchmarking behavior settings."""

    quality_threshold: float
    max_hallucination_risk: float
    deterministic_seed: int
    gemma_temperature: float
    vram_release_threshold_mb: float


@dataclass(frozen=True)
class WorkerConfig:
    """Background worker limits."""

    export_thread_pool_size: int


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""

    app_name: str
    app_version: str
    logging: LoggingConfig
    image: ImageConfig
    hardware: HardwareConfig
    paths: PathsConfig
    workers: WorkerConfig
    competition: CompetitionConfig
