"""Scene context builder interface."""

from typing import Protocol

from core.contracts.analysis import ActivityHints, AttributeSet, SceneContext, SceneGraph


class ISceneContextBuilder(Protocol):
    """Aggregate analysis outputs into scene context."""

    def build(
        self,
        graph: SceneGraph,
        attributes: AttributeSet,
        activities: ActivityHints,
    ) -> SceneContext:
        """Return unified scene context from scene graph artifacts."""
        ...
