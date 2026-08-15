"""Caption quality evaluator interface."""

from typing import Protocol

from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport


class ICaptionQualityEvaluator(Protocol):
    """Evaluate final caption quality against scene evidence."""

    def evaluate(self, caption: str, context: SceneContext) -> CaptionQualityReport:
        """Return internal quality report."""
        ...
