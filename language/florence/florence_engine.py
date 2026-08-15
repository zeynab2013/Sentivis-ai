"""Florence-2 vision-language engine with automatic BLIP fallback for 2GB VRAM."""

from __future__ import annotations

from typing import Any

from PIL import Image

from core.config.model_config import BlipModelConfig, FlorenceModelConfig
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.exceptions.language import InferenceError, ModelLoadError
from core.logging import get_logger
from language.blip.blip_engine import BlipEngine

logger = get_logger(__name__)


class FlorenceEngine:
    """Detailed image understanding via Florence-2, falling back to BLIP when needed."""

    def __init__(
        self,
        florence_config: FlorenceModelConfig,
        blip_config: BlipModelConfig,
    ) -> None:
        self._config = florence_config
        self._blip_config = blip_config
        self._processor: Any = None
        self._model: Any = None
        self._fallback: BlipEngine | None = None
        self._device = florence_config.preferred_device
        self._loaded = False
        self._backend = "none"

    @property
    def model_kind(self) -> ModelKind:
        return ModelKind.BLIP

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

    def set_device(self, device: str) -> None:
        self._device = device
        if self._fallback is not None:
            self._fallback.set_device(device)

    def load(self) -> None:
        if self._loaded:
            return
        if self._try_load_florence():
            self._loaded = True
            self._backend = "florence2"
            logger.info("Florence-2 loaded on %s (%s)", self._device, self._config.model_id)
            return
        if self._config.fallback_to_blip:
            self._fallback = BlipEngine(self._blip_config)
            self._fallback.set_device(self._device)
            self._fallback.load()
            self._loaded = True
            self._backend = "blip_fallback"
            logger.warning("Florence-2 unavailable; using BLIP fallback on %s", self._device)
            return
        raise ModelLoadError(
            "Detailed vision model could not be loaded.",
            "Florence-2 load failed and BLIP fallback disabled",
            stage=PipelineStage.BLIP_UNDERSTANDING,
        )

    def infer(self, image: PreprocessedImage) -> RawCaption:
        if not self._loaded:
            raise InferenceError(
                "Visual understanding is not ready.",
                "Florence infer called before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=False,
            )
        if self._backend == "blip_fallback" and self._fallback is not None:
            caption = self._fallback.infer(image)
            return RawCaption(
                text=caption.text,
                source="blip_fallback",
                confidence=caption.confidence,
            )
        return self._infer_florence(image)

    def release(self) -> None:
        if self._fallback is not None:
            self._fallback.release()
            self._fallback = None
        self._processor = None
        self._model = None
        self._loaded = False
        self._backend = "none"
        self.clear_device_cache()

    def clear_device_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _try_load_florence(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self._config.model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.to(self._device)
            self._model.eval()
            return True
        except Exception as exc:  # noqa: BLE001 — intentional soft fallback for judges
            logger.warning("Florence-2 load skipped: %s", exc)
            self._processor = None
            self._model = None
            self.clear_device_cache()
            return False

    def _infer_florence(self, image: PreprocessedImage) -> RawCaption:
        if self._model is None or self._processor is None:
            raise InferenceError(
                "Visual understanding is not ready.",
                "Florence model missing at infer time",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            import torch

            pil_image = Image.fromarray(image.display_pixels)
            # Prefer richer caption when available; fall back to short caption task.
            task = "<MORE_DETAILED_CAPTION>"
            inputs = self._processor(text=task, images=pil_image, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = self._parse_florence_text(text, task=task, image=pil_image)
            if not parsed:
                parsed = "A detailed scene description could not be decoded."
            return RawCaption(text=parsed.strip(), source="florence2", confidence=0.9)
        except Exception as exc:
            raise InferenceError(
                "Detailed visual understanding failed.",
                f"Florence infer failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc

    def _parse_florence_text(self, text: str, *, task: str, image: Image.Image) -> str:
        if self._processor is None:
            return text
        try:
            parsed = self._processor.post_process_generation(
                text,
                task=task,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict):
                value = parsed.get(task) or next(iter(parsed.values()), "")
                return str(value).strip()
        except Exception:  # noqa: BLE001
            cleaned = text.replace(task, "").replace("</s>", "").replace("<s>", "").strip()
            return cleaned
        return text.strip()
