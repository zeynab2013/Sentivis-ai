"""Activity analyzer interface."""

from typing import Protocol

from core.contracts.analysis import ActivityHints, SceneGraph


class IActivityAnalyzer(Protocol):
    """Infer activities from scene graph."""

    def analyze(self, graph: SceneGraph) -> ActivityHints:
        """Return activity hints."""
        ...
