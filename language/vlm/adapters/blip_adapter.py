"""BLIP vision adapter (CPU / low-VRAM fallback)."""

from __future__ import annotations

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class BlipVisionAdapter(BaseVisionAdapter):
    """Adapter wrapping Hugging Face BLIP."""

    def __init__(self, model_id: str, preferred_device: str = "cpu") -> None:
        super().__init__(model_id, "blip", preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._processor = BlipProcessor.from_pretrained(self._model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "BLIP vision model could not be loaded.",
                f"BLIP adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError(
                "Vision model is not ready.",
                "BLIP describe before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        import torch

        pil = self._pil(image)
        inputs = self._processor(pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_length=120)
        text = self._processor.decode(output[0], skip_special_tokens=True).strip()
        return RawCaption(text=text, source="blip", confidence=0.82)

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        """Condition BLIP on the evidence package, not the bare image alone."""
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError(
                "Vision model is not ready.",
                "BLIP narrate before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        import torch

        pil = self._pil(image)
        prompt = self._prompts.narrate_prompt(self._adapter_name, understanding)
        inputs = self._processor(pil, prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_length=160)
        text = self._processor.decode(output[0], skip_special_tokens=True).strip()
        # BLIP often echoes the prompt prefix — keep the generated tail when present.
        if prompt[:40].lower() in text.lower():
            text = text[len(prompt) :].strip() if text.lower().startswith(prompt.lower()[:20]) else text
        if not text:
            text = self.describe(image).text
        return RawCaption(text=text, source="blip", confidence=0.8)
