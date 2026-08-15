"""End-to-end system workflow verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.container import DependencyContainer
from app.startup.orchestrator import StartupOrchestrator
from certification.pipeline_harness import build_test_orchestrator
from certification.pipeline_stubs import StubDetector, StubReasoning, StubVisionLanguage
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from services.export.export_manager import ExportManager


@dataclass(frozen=True)
class WorkflowCheck:
    """One end-to-end workflow verification step."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SystemVerificationReport:
    """Aggregated end-to-end workflow results."""

    checks: tuple[WorkflowCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class SystemVerifier:
    """Verifies startup through shutdown workflows without modifying frozen layers."""

    def run(self, workspace: Path | None = None) -> SystemVerificationReport:
        checks: list[WorkflowCheck] = []
        checks.append(self._verify_startup())
        checks.append(self._verify_container_build())
        checks.append(self._verify_model_discovery())
        checks.append(self._verify_image_loading(workspace))
        checks.append(self._verify_pipeline_execution(workspace))
        checks.append(self._verify_result_presentation(workspace))
        checks.append(self._verify_export_workflow(workspace))
        checks.append(self._verify_shutdown())
        return SystemVerificationReport(checks=tuple(checks))

    def _verify_startup(self) -> WorkflowCheck:
        result = StartupOrchestrator().run()
        passed = result.context.facade is not None and len(result.report.stages) == 8
        detail = f"{len(result.report.stages)} startup stages, {len(result.report.errors)} errors"
        return WorkflowCheck("application_startup", passed, detail)

    def _verify_container_build(self) -> WorkflowCheck:
        context = DependencyContainer().build(
            load_app_config(),
            load_model_config(),
            load_theme_config(),
            load_analysis_config(),
        )
        passed = context.release_info is not None and context.runtime_status.health_score >= 0
        return WorkflowCheck(
            "configuration_loading",
            passed,
            f"health_score={context.runtime_status.health_score}",
        )

    def _verify_model_discovery(self) -> WorkflowCheck:
        context = DependencyContainer().build(
            load_app_config(),
            load_model_config(),
            load_theme_config(),
            load_analysis_config(),
        )
        records = context.model_registry.records
        passed = len(records) == 3
        return WorkflowCheck("model_discovery", passed, f"{len(records)} models registered")

    def _verify_image_loading(self, workspace: Path | None) -> WorkflowCheck:
        from vision.validation.image_validator import ImageValidator

        image_path = self._create_sample_image(workspace)
        validator = ImageValidator(load_app_config())
        validated = validator.validate(image_path)
        passed = validated.width == 64 and validated.height == 64
        return WorkflowCheck("image_loading", passed, f"loaded {validated.width}x{validated.height}")

    def _verify_pipeline_execution(self, workspace: Path | None) -> WorkflowCheck:
        image_path = self._create_sample_image(workspace)
        orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
        result = orchestrator.analyze(PipelineRequest(image_path, AnalysisOptions(enable_gemma=True)))
        passed = bool(result.caption.text) and result.scene_context.object_count == 1
        return WorkflowCheck(
            "pipeline_execution",
            passed,
            f"caption={result.caption.text[:40]!r}",
        )

    def _verify_result_presentation(self, workspace: Path | None) -> WorkflowCheck:
        image_path = self._create_sample_image(workspace)
        orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
        result = orchestrator.analyze(PipelineRequest(image_path, AnalysisOptions(enable_gemma=True)))
        passed = result.quality_report.overall_quality > 0 and bool(result.caption.text)
        return WorkflowCheck(
            "result_presentation",
            passed,
            f"quality={result.quality_report.overall_quality:.2f}",
        )

    def _verify_export_workflow(self, workspace: Path | None) -> WorkflowCheck:
        image_path = self._create_sample_image(workspace)
        orchestrator = build_test_orchestrator(StubDetector(), StubVisionLanguage(), StubReasoning())
        result = orchestrator.analyze(PipelineRequest(image_path, AnalysisOptions(enable_gemma=True)))
        export_dir = (workspace or Path.cwd()) / "cert_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        md_path = export_dir / "certification_report.md"
        ExportManager().export(result, "md", md_path)
        passed = md_path.is_file() and "Sentivis AI" in md_path.read_text(encoding="utf-8")
        return WorkflowCheck("export_workflow", passed, str(md_path.name))

    def _verify_shutdown(self) -> WorkflowCheck:
        context = DependencyContainer().build(
            load_app_config(),
            load_model_config(),
            load_theme_config(),
            load_analysis_config(),
        )
        context.model_manager.release_all()
        passed = True
        return WorkflowCheck("application_shutdown", passed, "model_manager.release_all completed")

    @staticmethod
    def _create_sample_image(workspace: Path | None) -> Path:
        directory = workspace or Path.cwd() / "cert_tmp"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "cert_sample.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
        return path
