"""Vision-language model interface."""

from typing import Protocol

from core.contracts.analysis import SceneContext
from core.contracts.image import PreprocessedImage
from core.contracts.language import VisualObservations


class IVisionLanguageModel(Protocol):
    """Model-agnostic visual description from image plus structured context."""

    def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
        """Generate visual observations from image and scene evidence."""
        ...
