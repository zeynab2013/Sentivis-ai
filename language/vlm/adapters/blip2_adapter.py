"""BLIP-2 vision adapter (CPU-capable fallback above classic BLIP)."""

from __future__ import annotations

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class Blip2VisionAdapter(BaseVisionAdapter):
    """Adapter wrapping Hugging Face BLIP-2."""

    def __init__(self, model_id: str, preferred_device: str = "cpu") -> None:
        super().__init__(model_id, "blip2", preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self._processor = Blip2Processor.from_pretrained(self._model_id)
            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
            try:
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    self._model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                self._model.to(self._device)
            except Exception:
                self._device = "cpu"
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self._model.to("cpu")
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "BLIP-2 vision model could not be loaded.",
                f"BLIP-2 adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        return self._generate(image, None)

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        prompt = self._prompts.narrate_prompt(self._adapter_name, understanding)
        return self._generate(image, prompt)

    def _generate(self, image: PreprocessedImage, prompt: str | None) -> RawCaption:
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError(
                "Vision model is not ready.",
                "BLIP-2 before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            import torch

            pil = self._pil(image)
            if prompt:
                inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(self._device)
            else:
                inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=180)
            text = self._processor.decode(output[0], skip_special_tokens=True).strip()
            return RawCaption(text=text, source="blip2", confidence=0.84)
        except Exception as exc:
            raise InferenceError(
                "BLIP-2 inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
