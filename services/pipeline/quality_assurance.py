"""Post-run caption quality assurance gate."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.app_config import CompetitionConfig
from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport, RefinedCaption
from core.logging import get_logger
from language.validation.caption_validator import CaptionEvidenceValidator

logger = get_logger(__name__)


@dataclass(frozen=True)
class QualityAssuranceResult:
    """Outcome of post-run caption validation."""

    passed: bool
    issues: tuple[str, ...]
    rejected_caption: bool


class PipelineQualityAssurance:
    """Verifies caption fidelity against structured scene evidence."""

    def __init__(self, config: CompetitionConfig) -> None:
        self._config = config
        self._validator = CaptionEvidenceValidator()

    def evaluate(
        self,
        caption: RefinedCaption,
        context: SceneContext,
        quality_report: CaptionQualityReport,
        *,
        strict: bool,
    ) -> QualityAssuranceResult:
        issues: list[str] = []
        threshold = self._config.quality_threshold if strict else 0.45
        max_hallucination = self._config.max_hallucination_risk if strict else 0.5

        unsupported = self._validator.unsupported_object_tokens(caption.text, context)
        if unsupported:
            issues.append(f"Unsupported object tokens: {', '.join(unsupported)}")

        if (
            quality_report.hallucination_risk is not None
            and quality_report.hallucination_risk > max_hallucination
        ):
            issues.append(
                f"Hallucination risk {quality_report.hallucination_risk:.2f} exceeds {max_hallucination:.2f}"
            )

        if quality_report.overall_quality < threshold:
            issues.append(
                f"Overall quality {quality_report.overall_quality:.2f} below threshold {threshold:.2f}"
            )

        if self._context_contradiction(caption.text, context):
            issues.append("Caption contradicts scene context markers")

        if self._graph_contradiction(caption.text, context):
            issues.append("Caption contradicts scene graph object set")

        passed = len(issues) == 0
        logger.info("QA gate passed=%s issues=%d strict=%s", passed, len(issues), strict)
        return QualityAssuranceResult(
            passed=passed,
            issues=tuple(issues),
            rejected_caption=not passed,
        )

    def _context_contradiction(self, caption: str, context: SceneContext) -> bool:
        lower = caption.lower()
        env = context.environment
        if env.indoor_outdoor == "indoor" and "outdoor" in lower and "indoor" not in lower:
            return True
        if env.indoor_outdoor == "outdoor" and "indoor" in lower and "outdoor" not in lower:
            return True
        return bool(env.crowd_level == "empty" and any(word in lower for word in ("crowd", "many people", "busy")))

    def _graph_contradiction(self, caption: str, context: SceneContext) -> bool:
        labels = {node.label.lower() for node in context.graph.nodes}
        if not labels:
            return False
        lower = caption.lower()
        mentioned = sum(1 for label in labels if label in lower)
        return context.object_count >= 2 and mentioned == 0 and len(lower.split()) > 8
