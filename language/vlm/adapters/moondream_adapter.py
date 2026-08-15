"""Moondream2 vision adapter for low-VRAM detailed understanding."""

from __future__ import annotations

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class MoondreamVisionAdapter(BaseVisionAdapter):
    """Adapter for vikhyatk/moondream2-style models."""

    def __init__(self, model_id: str, preferred_device: str = "cuda") -> None:
        super().__init__(model_id, "moondream", preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
            self._processor = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map={"": self._device} if str(self._device).startswith("cuda") else None,
                )
            except Exception:
                # Retry on CPU when CUDA OOM / device_map fails — never crash on VRAM.
                self._device = "cpu"
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            if not str(self._device).startswith("cuda"):
                self._model.to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "Moondream vision model could not be loaded.",
                f"Moondream adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        return self._ask(image, "Describe this image in detail.")

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        brief = (understanding.evidence_brief or "")[:800]
        question = (
            "Write one careful observational paragraph grounded in the image. "
            f"Verified evidence: {brief}"
        )
        return self._ask(image, question)

    def _ask(self, image: PreprocessedImage, question: str) -> RawCaption:
        if not self._loaded or self._model is None:
            raise InferenceError(
                "Vision model is not ready.",
                "Moondream before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            import torch

            pil = self._pil(image)
            with torch.no_grad():
                if hasattr(self._model, "answer_question"):
                    enc_image = self._model.encode_image(pil)
                    text = str(self._model.answer_question(enc_image, question, self._processor)).strip()
                else:
                    # Generic generate fallback for compatible casual-LM vision wrappers.
                    text = self.describe(image).text
            if not text:
                raise InferenceError(
                    "Moondream returned empty text.",
                    "empty response",
                    stage=PipelineStage.BLIP_UNDERSTANDING,
                    recoverable=True,
                )
            return RawCaption(text=text, source="moondream", confidence=0.86)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(
                "Moondream inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
