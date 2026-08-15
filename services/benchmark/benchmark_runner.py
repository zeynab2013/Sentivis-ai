"""Internal pipeline benchmarking facility."""

from __future__ import annotations

import json
import time
from pathlib import Path

from core.contracts.metrics import BenchmarkReport, BenchmarkSample
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from services.pipeline.orchestrator import PipelineOrchestrator


class BenchmarkRunner:
    """Runs repeatable pipeline benchmarks and exports reports."""

    def __init__(self, orchestrator: PipelineOrchestrator) -> None:
        self._orchestrator = orchestrator

    def run(
        self,
        image_paths: tuple[Path, ...],
        *,
        iterations: int = 3,
        competition_mode: bool = True,
    ) -> BenchmarkReport:
        if not image_paths:
            raise ValueError("At least one image path is required for benchmarking")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        samples: list[BenchmarkSample] = []
        cold_start_ms = 0.0
        warm_durations: list[float] = []
        peak_rams: list[float] = []
        peak_vrams: list[float] = []
        quality_scores: list[float] = []
        load_times: list[float] = []
        unload_times: list[float] = []

        for iteration in range(iterations):
            for index, image_path in enumerate(image_paths):
                cold = iteration == 0 and index == 0
                request = PipelineRequest(
                    image_path,
                    AnalysisOptions(enable_gemma=True, competition_mode=competition_mode),
                )
                started = time.perf_counter()
                result = self._orchestrator.analyze(request)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                metrics = result.metrics
                peak_ram = metrics.peak_ram_mb if metrics else 0.0
                peak_vram = metrics.peak_vram_mb if metrics else 0.0
                quality = metrics.caption_quality_score if metrics else result.quality_report.overall_quality
                objects = metrics.objects_detected if metrics else result.scene_context.object_count

                samples.append(
                    BenchmarkSample(
                        iteration=iteration + 1,
                        cold_start=cold,
                        total_duration_ms=elapsed_ms,
                        peak_ram_mb=peak_ram,
                        peak_vram_mb=peak_vram,
                        objects_detected=objects,
                        caption_quality_score=quality,
                    )
                )

                if cold:
                    cold_start_ms = elapsed_ms
                else:
                    warm_durations.append(elapsed_ms)
                peak_rams.append(peak_ram)
                peak_vrams.append(peak_vram)
                quality_scores.append(quality)

                if metrics:
                    for timing in metrics.model_timings:
                        load_times.append(timing.load_ms)
                        unload_times.append(timing.unload_ms)

        warm_avg = sum(warm_durations) / len(warm_durations) if warm_durations else cold_start_ms
        all_durations = [sample.total_duration_ms for sample in samples]
        avg_inference = sum(all_durations) / len(all_durations)
        total_minutes = sum(all_durations) / 60000.0
        throughput = len(samples) / total_minutes if total_minutes > 0 else 0.0

        return BenchmarkReport(
            image_count=len(image_paths),
            iteration_count=iterations,
            cold_start_ms=cold_start_ms,
            warm_start_avg_ms=warm_avg,
            avg_inference_ms=avg_inference,
            avg_peak_ram_mb=sum(peak_rams) / len(peak_rams),
            avg_peak_vram_mb=sum(peak_vrams) / len(peak_vrams),
            throughput_images_per_minute=throughput,
            avg_model_load_ms=sum(load_times) / len(load_times) if load_times else 0.0,
            avg_model_unload_ms=sum(unload_times) / len(unload_times) if unload_times else 0.0,
            samples=tuple(samples),
        )

    @staticmethod
    def export_report(report: BenchmarkReport, path: Path) -> None:
        payload = {
            "image_count": report.image_count,
            "iteration_count": report.iteration_count,
            "cold_start_ms": report.cold_start_ms,
            "warm_start_avg_ms": report.warm_start_avg_ms,
            "avg_inference_ms": report.avg_inference_ms,
            "avg_peak_ram_mb": report.avg_peak_ram_mb,
            "avg_peak_vram_mb": report.avg_peak_vram_mb,
            "throughput_images_per_minute": report.throughput_images_per_minute,
            "avg_model_load_ms": report.avg_model_load_ms,
            "avg_model_unload_ms": report.avg_model_unload_ms,
            "samples": [
                {
                    "iteration": sample.iteration,
                    "cold_start": sample.cold_start,
                    "total_duration_ms": sample.total_duration_ms,
                    "peak_ram_mb": sample.peak_ram_mb,
                    "peak_vram_mb": sample.peak_vram_mb,
                    "objects_detected": sample.objects_detected,
                    "caption_quality_score": sample.caption_quality_score,
                }
                for sample in report.samples
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
