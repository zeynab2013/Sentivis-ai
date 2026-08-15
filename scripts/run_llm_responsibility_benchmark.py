"""Benchmark heuristic-only vs split architecture on 20-image dataset."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from app.container import DependencyContainer  # noqa: E402
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config  # noqa: E402
from model_management.download.sources.ollama import detect_ollama  # noqa: E402
from scripts.run_real_world_evaluation import MANIFEST_PATH, evaluate_image  # noqa: E402

OUT = ROOT / "validation" / "real_world" / "llm_responsibility_benchmark.json"
LEGACY = ROOT / "validation" / "real_world" / "activity_benchmark_comparison.json"


@dataclass
class ModeMetrics:
    mode: str
    caption_quality: float
    hallucination_rate: float
    evidence_consistency: float
    semantic_score: float
    activity_reasoning: float
    duration_seconds: float


def _run(mode: str, manifest: dict) -> tuple[list[dict], ModeMetrics]:
    os.environ["SENTIVIS_SEMANTIC_MODE"] = mode
    os.environ["SENTIVIS_ACTIVITY_MODE"] = "heuristic"
    ctx = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    orchestrator = ctx.main_controller.pipeline._orchestrator  # noqa: SLF001
    started = time.perf_counter()
    rows: list[dict] = []
    for idx, item in enumerate(manifest["images"], start=1):
        print(f"[{mode}] {idx}/{len(manifest['images'])} {item['file_name']}", flush=True)
        rows.append(asdict(evaluate_image(orchestrator, item)))
    duration = time.perf_counter() - started
    ctx.model_manager.release_all()
    ctx.memory_manager.clear_gpu_cache()
    metrics = ModeMetrics(
        mode=mode,
        caption_quality=round(statistics.mean(r["caption_quality"] for r in rows), 3),
        hallucination_rate=round(statistics.mean(r["hallucination_rate"] for r in rows), 3),
        evidence_consistency=round(statistics.mean(r["evidence_consistency"] for r in rows), 3),
        semantic_score=round(statistics.mean(r["overall_semantic_score"] for r in rows), 3),
        activity_reasoning=round(statistics.mean(r["activity_reasoning"] for r in rows), 3),
        duration_seconds=round(duration, 1),
    )
    return rows, metrics


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ollama = detect_ollama()

    print("Benchmark: heuristic-only (semantic off)...")
    off_rows, off_metrics = _run("off", manifest)

    split_rows: list[dict] = []
    split_metrics: ModeMetrics | None = None
    if ollama.running:
        print("Benchmark: split architecture (heuristic activities + Ollama semantic)...")
        split_rows, split_metrics = _run("ollama", manifest)
    else:
        print("Ollama not running — split architecture benchmark skipped.")

    legacy_ollama_activity = None
    if LEGACY.exists():
        legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
        legacy_ollama_activity = legacy.get("ollama")

    payload = {
        "ollama_status": asdict(ollama),
        "heuristic_only": asdict(off_metrics),
        "split_architecture": asdict(split_metrics) if split_metrics else None,
        "legacy_ollama_activity_architecture": legacy_ollama_activity,
        "heuristic_only_evaluations": off_rows,
        "split_architecture_evaluations": split_rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
