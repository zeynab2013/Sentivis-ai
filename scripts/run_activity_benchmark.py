"""Compare heuristic vs Ollama activity reasoning on the 20-image real-world benchmark."""

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
from scripts.run_real_world_evaluation import (  # noqa: E402
    DATASET_DIR,
    MANIFEST_PATH,
    evaluate_image,
)

OUT_PATH = DATASET_DIR / "activity_benchmark_comparison.json"


@dataclass
class ModeSummary:
    mode: str
    activity_reasoning: float
    relationship_correctness: float
    caption_quality: float
    hallucination_rate: float
    overall_semantic_score: float
    duration_seconds: float


def _run_mode(mode: str, manifest: dict) -> tuple[list[dict], ModeSummary]:
    os.environ["SENTIVIS_ACTIVITY_MODE"] = mode
    ctx = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    orchestrator = ctx.main_controller.pipeline._orchestrator  # noqa: SLF001
    started = time.perf_counter()
    evaluations = []
    for idx, item in enumerate(manifest["images"], start=1):
        print(f"[{mode}] {idx}/{len(manifest['images'])} {item['file_name']}", flush=True)
        evaluations.append(asdict(evaluate_image(orchestrator, item)))
    duration = time.perf_counter() - started
    ctx.model_manager.release_all()
    ctx.memory_manager.clear_gpu_cache()

    summary = ModeSummary(
        mode=mode,
        activity_reasoning=round(statistics.mean(e["activity_reasoning"] for e in evaluations), 3),
        relationship_correctness=round(statistics.mean(e["relationship_correctness"] for e in evaluations), 3),
        caption_quality=round(statistics.mean(e["caption_quality"] for e in evaluations), 3),
        hallucination_rate=round(statistics.mean(e["hallucination_rate"] for e in evaluations), 3),
        overall_semantic_score=round(statistics.mean(e["overall_semantic_score"] for e in evaluations), 3),
        duration_seconds=round(duration, 1),
    )
    return evaluations, summary


def main() -> int:
    if not MANIFEST_PATH.exists():
        from scripts.build_real_world_dataset import build_dataset

        build_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ollama_status = detect_ollama()
    print(f"Ollama: installed={ollama_status.installed} running={ollama_status.running}")

    print("Running heuristic benchmark...")
    heuristic_evals, heuristic_summary = _run_mode("heuristic", manifest)

    ollama_evals: list[dict] = []
    ollama_summary: ModeSummary | None = None
    if ollama_status.running:
        print("Running Ollama benchmark...")
        ollama_evals, ollama_summary = _run_mode("ollama", manifest)
    else:
        print("Skipping Ollama benchmark — Ollama not running.")

    winner = "heuristic"
    if ollama_summary:
        heuristic_score = (
            heuristic_summary.activity_reasoning * 0.35
            + heuristic_summary.caption_quality * 0.25
            + (1.0 - heuristic_summary.hallucination_rate) * 0.20
            + heuristic_summary.relationship_correctness * 0.20
        )
        ollama_score = (
            ollama_summary.activity_reasoning * 0.35
            + ollama_summary.caption_quality * 0.25
            + (1.0 - ollama_summary.hallucination_rate) * 0.20
            + ollama_summary.relationship_correctness * 0.20
        )
        winner = "ollama" if ollama_score >= heuristic_score else "heuristic"

    payload = {
        "heuristic": asdict(heuristic_summary),
        "ollama": asdict(ollama_summary) if ollama_summary else None,
        "winner": winner,
        "heuristic_evaluations": heuristic_evals,
        "ollama_evaluations": ollama_evals,
        "ollama_status": asdict(ollama_status),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Comparison written to {OUT_PATH}")
    print(f"Recommended mode: {winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
