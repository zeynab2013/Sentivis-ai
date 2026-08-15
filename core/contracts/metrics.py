"""Pipeline metrics and benchmarking DTOs."""

from dataclasses import dataclass

from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage


@dataclass(frozen=True)
class StageMetric:
    """Timing and memory snapshot for one pipeline stage."""

    stage: PipelineStage
    duration_ms: float
    ram_used_mb: float
    vram_allocated_mb: float


@dataclass(frozen=True)
class ModelTimingMetric:
    """Load/unload timing for one model kind."""

    kind: ModelKind
    load_ms: float
    unload_ms: float
    gpu_released: bool


@dataclass(frozen=True)
class PipelineMetrics:
    """Diagnostics collected for one pipeline execution."""

    total_duration_ms: float
    peak_ram_mb: float
    peak_vram_mb: float
    stage_metrics: tuple[StageMetric, ...]
    model_timings: tuple[ModelTimingMetric, ...]
    objects_detected: int
    relationships_inferred: int
    activities_inferred: int
    scene_graph_nodes: int
    scene_graph_edges: int
    caption_quality_score: float
    recovery_events: int
    fallback_events: int
    competition_mode: bool
    qa_passed: bool
    vlm_executions: int = 0
    caption_generation_count: int = 1
    qa_count: int = 1


@dataclass(frozen=True)
class BenchmarkSample:
    """One benchmark iteration result."""

    iteration: int
    cold_start: bool
    total_duration_ms: float
    peak_ram_mb: float
    peak_vram_mb: float
    objects_detected: int
    caption_quality_score: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated benchmark run for export."""

    image_count: int
    iteration_count: int
    cold_start_ms: float
    warm_start_avg_ms: float
    avg_inference_ms: float
    avg_peak_ram_mb: float
    avg_peak_vram_mb: float
    throughput_images_per_minute: float
    avg_model_load_ms: float
    avg_model_unload_ms: float
    samples: tuple[BenchmarkSample, ...]
