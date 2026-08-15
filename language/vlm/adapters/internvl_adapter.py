"""InternVL vision adapter for high-VRAM systems."""

from __future__ import annotations

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class InternVLVisionAdapter(BaseVisionAdapter):
    """Optional InternVL adapter with graceful load failure."""

    def __init__(self, model_id: str, preferred_device: str = "cuda") -> None:
        super().__init__(model_id, "internvl", preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
            self._processor = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_id,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
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
                "InternVL vision model could not be loaded.",
                f"InternVL adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        return self._chat(image, self._prompts.describe_prompt(self._adapter_name))

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        return self._chat(image, self._prompts.narrate_prompt(self._adapter_name, understanding))

    def _chat(self, image: PreprocessedImage, prompt: str) -> RawCaption:
        if not self._loaded or self._model is None:
            raise InferenceError(
                "Vision model is not ready.",
                "InternVL chat before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            pil = self._pil(image)
            if hasattr(self._model, "chat"):
                response = self._model.chat(self._processor, pil, prompt, generation_config=None)
                text = response if isinstance(response, str) else str(response)
            else:
                raise RuntimeError("InternVL chat API unavailable on loaded model")
            return RawCaption(text=text.strip(), source="internvl", confidence=0.92)
        except Exception as exc:
            raise InferenceError(
                "InternVL inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
