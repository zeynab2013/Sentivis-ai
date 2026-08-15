"""Managed VLM with automatic selection and failover (implements IVisionLanguageModel)."""

from __future__ import annotations

from core.config.model_config import ModelConfig
from core.config.vlm_config import VlmSelectionConfig
from core.contracts.analysis import SceneContext
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption, VisualObservations
from core.contracts.reasoning import SceneUnderstanding
from core.logging import get_logger
from language.blip.observation_mapper import BlipObservationMapper
from language.vlm.adapters.base import BaseVisionAdapter
from language.vlm.adapters.factory import create_vision_model
from language.vlm.selector import VlmChoice, VlmSelector

logger = get_logger(__name__)


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "cuda out of memory",
            "cudnn_status_alloc_failed",
            "hip out of memory",
            "oom",
        )
    )


class ManagedVisionModel:
    """Selects, loads, and fails over VLM adapters while exposing understand/narrate."""

    def __init__(self, model_config: ModelConfig, vlm_config: VlmSelectionConfig) -> None:
        self._model_config = model_config
        self._vlm_config = vlm_config
        self._selector = VlmSelector(vlm_config)
        self._mapper = BlipObservationMapper()
        self._adapter: BaseVisionAdapter | None = None
        self._active_name = ""
        self._execution_count = 0

    @property
    def execution_count(self) -> int:
        """Successful VLM perception/inference calls since the last reset."""
        return self._execution_count

    def reset_execution_count(self) -> None:
        """Reset per-image VLM execution counter."""
        self._execution_count = 0

    def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
        caption = self._with_failover(lambda adapter: adapter.describe(image), purpose="perceive")
        return self._mapper.map(caption, context)

    def narrate(
        self,
        image: PreprocessedImage,
        understanding: SceneUnderstanding,
    ) -> RawCaption:
        return self._with_failover(lambda adapter: adapter.narrate(image, understanding), purpose="narrate")

    def release(self) -> None:
        if self._adapter is not None:
            self._adapter.release()
            self._adapter = None
            self._active_name = ""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _with_failover(self, operation: object, *, purpose: str = "perceive") -> RawCaption:
        from collections.abc import Callable

        if not callable(operation):
            raise TypeError("VLM operation must be callable")
        call: Callable[[BaseVisionAdapter], RawCaption] = operation
        primary = self._selector.select()
        chain = (primary, *self._selector.fallback_chain(primary.adapter_name))
        last_error: Exception | None = None
        for choice in chain:
            try:
                adapter = self._ensure_adapter(choice.adapter_name, choice.model_id)
                result = call(adapter)
                self._execution_count += 1
                logger.info(
                    "VLM inference via %s purpose=%s | VLM execution count: %d",
                    choice.adapter_name,
                    purpose,
                    self._execution_count,
                )
                if purpose != "perceive":
                    logger.warning(
                        "Unexpected extra VLM call purpose=%s — perception should run once per image",
                        purpose,
                    )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "VLM adapter %s failed purpose=%s: %s",
                    choice.adapter_name,
                    purpose,
                    exc,
                )
                # Same adapter on CPU before abandoning to the next model.
                if _is_oom_error(exc) and not str(self._model_config.blip.preferred_device).startswith("cpu"):
                    try:
                        self.release()
                        adapter = self._ensure_adapter(
                            choice.adapter_name,
                            choice.model_id,
                            force_device="cpu",
                        )
                        result = call(adapter)
                        self._execution_count += 1
                        logger.info(
                            "VLM inference via %s on CPU after OOM purpose=%s | VLM execution count: %d",
                            choice.adapter_name,
                            purpose,
                            self._execution_count,
                        )
                        logger.warning("VLM retry reason: CUDA OOM — recovered on CPU")
                        return result
                    except Exception as cpu_exc:  # noqa: BLE001
                        last_error = cpu_exc
                        logger.warning(
                            "VLM adapter %s CPU recovery failed: %s",
                            choice.adapter_name,
                            cpu_exc,
                        )
                self.release()
                continue
        raise RuntimeError(f"All VLM adapters failed: {last_error}")

    def _ensure_adapter(
        self,
        adapter_name: str,
        model_id: str,
        *,
        force_device: str | None = None,
    ) -> BaseVisionAdapter:
        if (
            self._adapter is not None
            and self._active_name == adapter_name
            and force_device is None
        ):
            return self._adapter
        self.release()
        device = force_device or self._model_config.blip.preferred_device
        adapter = create_vision_model(VlmChoice(adapter_name, model_id, "runtime"), device)
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    adapter.set_device("cpu")
            except ImportError:
                adapter.set_device("cpu")
        try:
            adapter.load()
        except Exception as exc:
            if _is_oom_error(exc) and not str(device).startswith("cpu"):
                logger.warning("VLM load OOM on %s — retrying %s on CPU", device, adapter_name)
                self.release()
                adapter = create_vision_model(VlmChoice(adapter_name, model_id, "runtime"), "cpu")
                adapter.set_device("cpu")
                adapter.load()
            else:
                raise
        self._adapter = adapter
        self._active_name = adapter_name
        return adapter
