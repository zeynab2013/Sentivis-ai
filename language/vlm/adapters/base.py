"""Shared adapter utilities."""

from __future__ import annotations

from typing import Any

from PIL import Image

from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.logging import get_logger
from language.vlm.prompt_builders import VisionPromptBuilder

logger = get_logger(__name__)


class BaseVisionAdapter:
    """Common lifecycle helpers for VLM adapters."""

    def __init__(self, model_id: str, adapter_name: str, preferred_device: str) -> None:
        self._model_id = model_id
        self._adapter_name = adapter_name
        self._device = preferred_device
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._prompts = VisionPromptBuilder()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def adapter_name(self) -> str:
        return self._adapter_name

    def set_device(self, device: str) -> None:
        self._device = device

    def load(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _pil(self, image: PreprocessedImage) -> Image.Image:
        return Image.fromarray(image.display_pixels)

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        """Default narrate: describe image then rely on external grounding."""
        return self.describe(image)

    def describe(self, image: PreprocessedImage) -> RawCaption:
        raise NotImplementedError
