"""Prompt builder interface."""

from typing import Protocol

from core.contracts.analysis import SceneContext
from core.contracts.language import Prompt, VisualObservations


class IPromptBuilder(Protocol):
    """Build reasoning prompt from scene context and BLIP observations."""

    def build(
        self,
        context: SceneContext,
        observations: VisualObservations | None = None,
    ) -> Prompt:
        """Return structured deterministic prompt."""
        ...
