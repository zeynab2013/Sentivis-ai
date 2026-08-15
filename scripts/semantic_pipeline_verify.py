"""Run semantic pipeline verification on runtime_verify_sample.png and capture metrics."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from app.container import DependencyContainer  # noqa: E402
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config  # noqa: E402
from core.contracts.pipeline import AnalysisOptions, PipelineRequest  # noqa: E402
from core.utils.paths import project_root  # noqa: E402
from ui.formatters.result_formatters import (  # noqa: E402
    format_execution_metrics,
    format_quality_report,
)


@dataclass
class SemanticRunReport:
    image_path: str = ""
    duration_seconds: float = 0.0
    caption: str = ""
    caption_sources: list[str] = field(default_factory=list)
    object_count: int = 0
    relationship_count: int = 0
    activity_count: int = 0
    environment_setting: str = ""
    environment_scene_type: str = ""
    spatial_summary: str = ""
    quality_overall: float = 0.0
    evidence_consistency: float = 0.0
    relationship_coverage: float = 0.0
    activity_coverage: float = 0.0
    object_coverage: float = 0.0
    stage_timings: list[dict[str, float | str]] = field(default_factory=list)
    invalid_inside_relations: list[str] = field(default_factory=list)


def _find_invalid_inside(result) -> list[str]:
    graph = result.scene_context.graph
    labels = {node.index: node.label.lower() for node in graph.nodes}
    never_containers = {
        "sports ball",
        "tennis racket",
        "person",
        "people",
        "man",
        "woman",
        "child",
    }
    issues: list[str] = []
    for relation in graph.relations:
        if relation.relation_type != "inside":
            continue
        inner = labels.get(relation.subject_index, "")
        outer = labels.get(relation.object_index, "")
        if inner in {"person", "people", "man", "woman", "child"} and outer in never_containers:
            issues.append(f"{inner} inside {outer}")
        if inner in {"person", "people", "man", "woman", "child"} and outer in {
            "person",
            "people",
            "man",
            "woman",
            "child",
        }:
            issues.append(f"{inner} inside {outer}")
    return issues


def run_semantic_verification(image_path: Path | None = None) -> SemanticRunReport:
    report = SemanticRunReport()
    sample = image_path or (project_root() / "runtime_verify_sample.png")
    report.image_path = str(sample)

    ctx = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    orchestrator = ctx.main_controller.pipeline._orchestrator  # noqa: SLF001
    request = PipelineRequest(sample, AnalysisOptions(enable_gemma=True))

    started = time.perf_counter()
    result = orchestrator.analyze(request)
    report.duration_seconds = time.perf_counter() - started

    report.caption = result.caption.text
    report.caption_sources = list(result.caption.sources)
    report.object_count = result.scene_context.object_count
    report.relationship_count = len(result.scene_context.graph.relations)
    report.activity_count = len(result.scene_context.activities.activities)
    report.environment_setting = result.scene_context.environment.setting
    report.environment_scene_type = result.scene_context.environment.scene_type
    report.spatial_summary = result.scene_context.spatial_summary
    report.quality_overall = result.quality_report.overall_quality
    report.evidence_consistency = result.quality_report.evidence_consistency
    report.relationship_coverage = result.quality_report.relationship_coverage
    report.activity_coverage = result.quality_report.activity_coverage
    report.object_coverage = result.quality_report.object_coverage
    report.stage_timings = [
        {"stage": item.stage.name, "duration_ms": item.duration_ms}
        for item in result.metrics.stage_metrics
    ]
    report.invalid_inside_relations = _find_invalid_inside(result)

    out = project_root() / "semantic_pipeline_report.json"
    payload = asdict(report)
    payload["quality_report_text"] = format_quality_report(result)
    payload["execution_metrics_text"] = format_execution_metrics(result)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ctx.model_manager.release_all()
    ctx.memory_manager.clear_gpu_cache()
    print(json.dumps(payload, indent=2))
    return report


def main() -> int:
    run_semantic_verification()
    return 0


if __name__ == "__main__":
    sys.exit(main())
