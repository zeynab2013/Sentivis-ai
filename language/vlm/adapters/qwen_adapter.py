"""Qwen2.5-VL vision adapter."""

from __future__ import annotations

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class QwenVisionAdapter(BaseVisionAdapter):
    """Optional Qwen2.5-VL adapter — soft-fails when package/weights unavailable."""

    def __init__(self, model_id: str, preferred_device: str = "cuda") -> None:
        super().__init__(model_id, "qwen", preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_id,
                trust_remote_code=True,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "Qwen vision model could not be loaded.",
                f"Qwen adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        return self._generate(image, self._prompts.describe_prompt(self._adapter_name))

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        return self._generate(image, self._prompts.narrate_prompt(self._adapter_name, understanding))

    def _generate(self, image: PreprocessedImage, prompt: str) -> RawCaption:
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError(
                "Vision model is not ready.",
                "Qwen generate before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            import torch

            pil = self._pil(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[pil], return_tensors="pt").to(self._device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=220)
            trimmed = output[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            return RawCaption(text=decoded, source="qwen", confidence=0.91)
        except Exception as exc:
            raise InferenceError(
                "Qwen inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
