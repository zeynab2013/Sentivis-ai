"""Benchmark runner tests."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from certification.pipeline_stubs import StubDetector, StubReasoning, StubVisionLanguage
from services.benchmark.benchmark_runner import BenchmarkRunner
from tests.support.pipeline_harness import build_test_orchestrator


def test_benchmark_runner_exports_report(tmp_path: Path) -> None:
    image_path = tmp_path / "bench.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_path)
    orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
    runner = BenchmarkRunner(orchestrator)
    report = runner.run((image_path,), iterations=2, competition_mode=True)
    export_path = tmp_path / "benchmark.json"
    BenchmarkRunner.export_report(report, export_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["iteration_count"] == 2
    assert payload["avg_inference_ms"] > 0.0
    assert len(payload["samples"]) == 2
