"""Image validator interface."""

from pathlib import Path
from typing import Protocol

from core.contracts.image import ValidatedImage


class IImageValidator(Protocol):
    """Contract for image validation before pipeline processing."""

    def validate(self, path: Path) -> ValidatedImage:
        """Validate and load an image from disk."""
        ...
