"""Reasoning model interface."""

from typing import Protocol

from core.contracts.analysis import SceneContext
from core.contracts.language import Prompt, RawCaption


class IReasoningModel(Protocol):
    """Model-agnostic text reasoning over structured evidence."""

    def reason(self, prompt: Prompt, context: SceneContext) -> RawCaption:
        """Generate reasoned caption from prompt and scene context."""
        ...
