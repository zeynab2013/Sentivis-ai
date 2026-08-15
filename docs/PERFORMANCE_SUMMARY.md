# Performance Summary

**Target hardware:** Windows 11, Python 3.10.11, NVIDIA GPU (2 GB VRAM), 8–16 GB RAM, mid-range CPU

---

## Design Targets

| Metric | Policy |
|--------|--------|
| GPU memory | One model resident at a time; release + cache clear between stages |
| VRAM headroom | Release verified against 64 MB threshold (configurable) |
| CPU fallback | Enabled when GPU load fails (`cpu_fallback_enabled`) |
| Pipeline timeout | 600 s default (`pipeline_timeout_seconds`) |
| RAM monitoring | Peak tracked per run via `MemoryManager` |

---

## Measured Benchmarks (Stub Integration)

Automated benchmarks use stub models (no real weight loading) to validate the metrics and benchmarking infrastructure:

| Metric | Typical stub run |
|--------|------------------|
| Full pipeline (64×64 image) | ~100–500 ms |
| Stage count recorded | 13+ stages |
| Peak RAM | Process RSS logged per stage |
| Benchmark export | JSON via `BenchmarkRunner.export_report()` |

Real-model performance on 2 GB VRAM depends on weight availability and is documented under Known AI Limitations. The benchmarking facility is ready for on-device measurement.

---

## Model Timing

Each GPU stage records via `ModelCycleTiming`:

- Load duration (ms)
- Unload duration (ms)
- GPU release verification (bool)

Aggregated in `PipelineMetrics.model_timings` and benchmark reports.

---

## Throughput

`BenchmarkRunner` computes `throughput_images_per_minute` from repeated iterations. Competition mode enables reproducible comparison runs.

---

## Diagnostics

Every `PipelineResult.metrics` includes:

- `total_duration_ms`
- `peak_ram_mb`, `peak_vram_mb`
- Per-stage `StageMetric` entries
- Recovery and fallback event counts

Export JSON includes metrics block when using `ExportManager`.
