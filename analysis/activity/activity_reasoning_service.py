"""Heuristic low-level activity detection (benchmark / pipeline default)."""

from __future__ import annotations

import os

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.activity.minimal_activity_fallback import MinimalActivityFallback
from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import ActivityHints, SceneGraph
from core.logging import get_logger

logger = get_logger(__name__)


class ActivityReasoningService:
    """Low-level activity detection — heuristics only; never delegates activities to LLM."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config
        self._heuristic = HeuristicActivityAnalyzer(analysis_config)
        self._fallback = MinimalActivityFallback(analysis_config)

    def detect_activities(self, graph: SceneGraph, *, mode: str | None = None) -> ActivityHints:
        """Return heuristic activity hints from scene graph evidence."""
        resolved = (mode or os.environ.get("SENTIVIS_ACTIVITY_MODE") or "heuristic").lower()
        if resolved == "minimal":
            return self._fallback.analyze(graph)
        return self._heuristic.analyze(graph)
