"""Minimal evidence-only activity fallback when LLM reasoning is unavailable."""

from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import ActivityHints, SceneGraph
from core.logging import get_logger

logger = get_logger(__name__)


class MinimalActivityFallback:
    """Returns only graph-observable activities without hardcoded scene rules."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config

    def analyze(self, graph: SceneGraph) -> ActivityHints:
        # Prefer empty over placeholder activities like "people present".
        _ = graph
        logger.debug("Minimal activity fallback returned no verified activities")
        return ActivityHints(activities=(), confidence=0.0)
