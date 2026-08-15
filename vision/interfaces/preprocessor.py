"""Image preprocessor interface."""

from typing import Protocol

from core.contracts.image import PreprocessedImage, ValidatedImage


class IImagePreprocessor(Protocol):
    """Contract for preparing validated images for inference."""

    def preprocess(self, image: ValidatedImage) -> PreprocessedImage:
        """Produce display and inference pixel buffers."""
        ...
