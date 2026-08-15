"""Gemma reasoning engine."""

from __future__ import annotations

from typing import Any

from core.config.model_config import GemmaModelConfig
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.contracts.language import Prompt, RawCaption
from core.exceptions.language import InferenceError, ModelLoadError
from core.logging import get_logger

logger = get_logger(__name__)


class GemmaEngine:
    """Manages Gemma model lifecycle and text generation."""

    def __init__(self, config: GemmaModelConfig) -> None:
        self._config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = config.preferred_device
        self._loaded = False

    @property
    def model_kind(self) -> ModelKind:
        return ModelKind.GEMMA

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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from model_management.auth import resolve_hf_token

            token = resolve_hf_token()
            tokenizer_cls: Any = AutoTokenizer
            self._tokenizer = tokenizer_cls.from_pretrained(self._config.model_id, token=token)
            load_kwargs: dict[str, object] = {"token": token}
            if self._config.quantization == "int4" and self._device.startswith("cuda"):
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_config_cls: Any = BitsAndBytesConfig
                    load_kwargs["quantization_config"] = bnb_config_cls(load_in_4bit=True)
                except ImportError:
                    logger.warning("bitsandbytes unavailable; loading Gemma without INT4")
            model_cls: Any = AutoModelForCausalLM
            try:
                self._model = model_cls.from_pretrained(
                    self._config.model_id,
                    **load_kwargs,
                )
                if "quantization_config" not in load_kwargs:
                    self._model.to(self._device)
            except (OSError, RuntimeError, ValueError) as cuda_exc:
                # Never terminate on VRAM — continue reasoning on CPU.
                message = str(cuda_exc).lower()
                if self._device.startswith("cuda") and any(
                    token in message for token in ("out of memory", "cuda", "oom", "hip")
                ):
                    logger.warning("Gemma CUDA load failed (%s) — falling back to CPU", cuda_exc)
                    self._device = "cpu"
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    self._model = model_cls.from_pretrained(self._config.model_id, token=token)
                    self._model.to("cpu")
                else:
                    raise
            self._model.eval()
            self._loaded = True
            logger.info("Gemma loaded on %s", self._device)
        except (OSError, RuntimeError) as exc:
            raise ModelLoadError(
                "Reasoning model could not be loaded.",
                f"Gemma load failed: {exc}",
                stage=PipelineStage.GEMMA_REASONING,
            ) from exc

    def infer(self, prompt: Prompt) -> RawCaption:
        if not self._model or not self._tokenizer:
            raise InferenceError(
                "Reasoning model is not ready.",
                "Gemma infer called before load",
                stage=PipelineStage.GEMMA_REASONING,
                recoverable=False,
            )
        try:
            import torch

            from services.pipeline.competition_context import is_active, seed

            if is_active():
                torch.manual_seed(seed())
            temperature = 0.0 if is_active() else self._config.temperature
            full_prompt = f"{prompt.system}\n\n{prompt.user}"
            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            device = self._device if self._device != "cpu" else "cpu"
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            return RawCaption(text=text.strip(), source="gemma", confidence=0.8)
        except RuntimeError as exc:
            raise InferenceError(
                "Reasoning failed during caption generation.",
                f"Gemma inference error: {exc}",
                stage=PipelineStage.GEMMA_REASONING,
                recoverable=True,
            ) from exc

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("Gemma model released")

    def clear_device_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

