"""Collects pipeline execution metrics for diagnostics and benchmarking."""

from __future__ import annotations

import time

from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport
from core.contracts.metrics import ModelTimingMetric, PipelineMetrics, StageMetric
from services.memory.memory_manager import MemoryManager


class PipelineMetricsCollector:
    """Records per-stage and per-run pipeline metrics."""

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory = memory_manager
        self._started_at = 0.0
        self._stage_metrics: list[StageMetric] = []
        self._model_timings: list[ModelTimingMetric] = []
        self._recovery_events = 0
        self._fallback_events = 0
        self._competition_mode = False

    def begin_run(self, *, competition_mode: bool) -> None:
        self._started_at = time.perf_counter()
        self._stage_metrics.clear()
        self._model_timings.clear()
        self._recovery_events = 0
        self._fallback_events = 0
        self._competition_mode = competition_mode
        self._memory.reset_peaks()

    def record_stage(self, stage: PipelineStage, duration_ms: float) -> None:
        snap = self._memory.snapshot()
        self._stage_metrics.append(
            StageMetric(
                stage=stage,
                duration_ms=max(duration_ms, 0.1),
                ram_used_mb=snap.ram_used_mb,
                vram_allocated_mb=snap.vram_allocated_mb,
            )
        )

    def record_model_timing(
        self,
        kind: ModelKind,
        load_ms: float,
        unload_ms: float,
        gpu_released: bool,
    ) -> None:
        self._model_timings.append(
            ModelTimingMetric(
                kind=kind,
                load_ms=load_ms,
                unload_ms=unload_ms,
                gpu_released=gpu_released,
            )
        )

    def record_recovery(self) -> None:
        self._recovery_events += 1

    def record_fallback(self) -> None:
        self._fallback_events += 1

    def finalize(
        self,
        scene_context: SceneContext,
        quality_report: CaptionQualityReport,
        *,
        qa_passed: bool,
        vlm_executions: int = 0,
        caption_generation_count: int = 1,
        qa_count: int = 1,
    ) -> PipelineMetrics:
        total_ms = (time.perf_counter() - self._started_at) * 1000.0
        graph = scene_context.graph
        from analysis.relationships.relation_metrics import count_meaningful_relations

        return PipelineMetrics(
            total_duration_ms=total_ms,
            peak_ram_mb=self._memory.peak_ram_mb,
            peak_vram_mb=self._memory.peak_vram_mb,
            stage_metrics=tuple(self._stage_metrics),
            model_timings=tuple(self._model_timings),
            objects_detected=len(graph.nodes),
            # Count the same semantic relations shown in UI / used downstream.
            relationships_inferred=count_meaningful_relations(graph),
            activities_inferred=len(scene_context.activities.activities),
            scene_graph_nodes=len(graph.nodes),
            scene_graph_edges=len(graph.relations),
            caption_quality_score=quality_report.overall_quality,
            recovery_events=self._recovery_events,
            fallback_events=self._fallback_events,
            competition_mode=self._competition_mode,
            qa_passed=qa_passed,
            vlm_executions=max(0, int(vlm_executions)),
            caption_generation_count=max(0, int(caption_generation_count)),
            qa_count=max(0, int(qa_count)),
        )
