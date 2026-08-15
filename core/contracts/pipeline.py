"""Pipeline request and result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.constants.pipeline_stages import PipelineStage
from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport, RefinedCaption
from core.contracts.metrics import PipelineMetrics
from core.contracts.verified_evidence import VerifiedSceneEvidence

if TYPE_CHECKING:
    from core.contracts.image_quality import ImageQualityReport

@dataclass(frozen=True)
class AnalysisOptions:
    """User-selected analysis options."""

    enable_gemma: bool = True
    enable_export_cache: bool = True
    competition_mode: bool = False
    enable_enhancement: bool = True
    enable_super_resolution: bool = True
    enable_sam2: bool = True

@dataclass(frozen=True)
class PipelineRequest:
    """Input to the analysis pipeline."""

    image_path: Path
    options: AnalysisOptions


@dataclass(frozen=True)
class StageProgress:
    """Progress event emitted during pipeline execution."""

    stage: PipelineStage
    percent: float
    message: str
    device: str


@dataclass(frozen=True)
class PipelineResult:
    """Complete output of a successful or partially successful pipeline run."""

    request: PipelineRequest
    scene_context: SceneContext
    caption: RefinedCaption
    quality_report: CaptionQualityReport
    metrics: PipelineMetrics
    qa_passed: bool
    stages_completed: tuple[PipelineStage, ...]
    warnings: tuple[str, ...]
    image_quality: ImageQualityReport | None = None
    enhanced_preview_path: Path | None = None
    # Cached caption translations: (("fa", "..."), ("de", "..."), ...)
    caption_translations: tuple[tuple[str, str], ...] = ()
    # Compact evidence for Vision Assistant (no VLM re-run).
    evidence_brief: str = ""
    ocr_snippets: tuple[str, ...] = ()
    initial_vlm_calls: int = 0
    # Canonical verified evidence — Caption + QA must consume this, not raw SceneContext.
    verified_evidence: VerifiedSceneEvidence | None = None