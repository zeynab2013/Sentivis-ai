"""BLIP vision-language engine."""

from __future__ import annotations

from typing import Any

from PIL import Image

from core.config.model_config import BlipModelConfig
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.exceptions.language import InferenceError, ModelLoadError
from core.logging import get_logger

logger = get_logger(__name__)


class BlipEngine:
    """Manages BLIP model lifecycle and caption generation."""

    def __init__(self, config: BlipModelConfig) -> None:
        self._config = config
        self._processor: Any = None
        self._model: Any = None
        self._device = config.preferred_device
        self._loaded = False

    @property
    def model_kind(self) -> ModelKind:
        return ModelKind.BLIP

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._processor = BlipProcessor.from_pretrained(self._config.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self._config.model_id)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info("BLIP loaded on %s", self._device)
        except (OSError, RuntimeError) as exc:
            raise ModelLoadError(
                "Visual understanding model could not be loaded.",
                f"BLIP load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def infer(self, image: PreprocessedImage) -> RawCaption:
        if not self._model or not self._processor:
            raise InferenceError(
                "Visual understanding is not ready.",
                "BLIP infer called before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=False,
            )
        try:
            import torch

            pil_image = Image.fromarray(image.display_pixels)
            inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_length=self._config.max_length)
            text = self._processor.decode(output[0], skip_special_tokens=True)
            return RawCaption(text=text.strip(), source="blip", confidence=0.85)
        except RuntimeError as exc:
            raise InferenceError(
                "Visual understanding failed during analysis.",
                f"BLIP inference error: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        logger.info("BLIP model released")

    def clear_device_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

