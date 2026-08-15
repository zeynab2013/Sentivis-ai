"""Automatic vision-language model selection from hardware capabilities."""



from __future__ import annotations



import json

import urllib.error

import urllib.request

from dataclasses import dataclass



from core.config.vlm_config import VlmSelectionConfig

from core.logging import get_logger



logger = get_logger(__name__)





@dataclass(frozen=True)

class VlmChoice:

    """Selected adapter + configured model id."""



    adapter_name: str

    model_id: str

    reason: str





class VlmSelector:

    """Choose the strongest compatible VLM for 2GB-class hardware.



    Preferred order (accuracy first; slow is acceptable):

    1. Gemma 3 Vision via Ollama (out-of-process — best for low VRAM)

    2. Florence-2 Base Fine-Tuned

    3. Florence-2 Base

    4. Moondream2

    5. BLIP2

    6. BLIP Large

    """



    def __init__(self, config: VlmSelectionConfig) -> None:

        self._config = config



    def select(self) -> VlmChoice:

        preferred = self._config.preferred_adapter.strip()
        # Competition policy: Qwen/InternVL stay out of automatic selection.
        if preferred in {"qwen", "internvl"}:
            logger.warning(
                "Preferred adapter %s disabled for competition; using florence_base",
                preferred,
            )
            preferred = "florence_base"
        if preferred:

            return self._choice_for_adapter(preferred)



        if not self._config.auto_select:

            return self._choice_for_adapter("florence_base")



        vram_gb = self._detect_vram_gb()

        # Perception ladder: Florence/BLIP family. Gemma 3 4B is reserved for
        # text semantic synthesis + caption translation (not multimodal describe).
        ladder: tuple[tuple[str, str, float, bool], ...] = (
            ("florence_base", self._config.model_ids.florence_base, self._config.min_vram_florence_base_gb, False),
            ("florence_plain", self._config.model_ids.florence_plain, self._config.min_vram_florence_plain_gb, False),
            ("moondream", self._config.model_ids.moondream, self._config.min_vram_moondream_gb, False),
            ("blip2", self._config.model_ids.blip2, self._config.min_vram_blip2_gb, False),
            ("blip", self._config.model_ids.blip, 0.0, False),
        )

        for name, model_id, need, requires_ollama in ladder:

            if requires_ollama and not self._ollama_model_available(model_id):

                continue

            # Soft +0.25GB margin: try the stronger model when close; OOM→CPU handles failure.

            if vram_gb <= 0.0:

                # CPU path: Ollama first, then Florence/BLIP (slow OK).

                if requires_ollama or name in {"florence_base", "florence_plain", "moondream", "blip2", "blip"}:

                    choice = VlmChoice(name, model_id, f"CPU → {name} (quality-first)")

                    logger.info("VLM selected: %s (%s)", choice.adapter_name, choice.reason)

                    return choice

                continue

            if vram_gb + 0.25 >= need:

                choice = VlmChoice(name, model_id, f"{vram_gb:.1f}GB → {name} (quality-first)")

                logger.info("VLM selected: %s (%s)", choice.adapter_name, choice.reason)

                return choice



        choice = VlmChoice("blip", self._config.model_ids.blip, f"{vram_gb:.1f}GB → BLIP fallback")

        logger.info("VLM selected: %s (%s)", choice.adapter_name, choice.reason)

        return choice



    def fallback_chain(self, failed: str) -> tuple[VlmChoice, ...]:
        # Fast perception failover. Skip heavy BLIP-2/Moondream retries that
        # inflate wall-time when Florence is incompatible. Gemma multimodal is
        # last-resort perception only; primary Gemma 3 4B role remains text
        # semantic synthesis + caption translation.
        order = (
            "florence_base",
            "florence_plain",
            "gemma_vision",
            "blip",
        )
        start = order.index(failed) + 1 if failed in order else 0
        return tuple(self._choice_for_adapter(name) for name in order[start:])



    def _choice_for_adapter(self, adapter_name: str) -> VlmChoice:

        ids = self._config.model_ids

        mapping = {

            "gemma_vision": ids.gemma_vision,

            "florence_base": ids.florence_base,

            "florence_plain": ids.florence_plain,

            "florence_large": ids.florence_large,

            "moondream": ids.moondream,

            "blip2": ids.blip2,

            "blip": ids.blip,

            "qwen": ids.qwen,

            "internvl": ids.internvl,

        }

        normalized = adapter_name.lower().strip()

        if normalized not in mapping:

            normalized = "florence_base"

        return VlmChoice(normalized, mapping[normalized], f"preferred={normalized}")



    @staticmethod

    def _detect_vram_gb() -> float:

        try:

            import torch



            if not torch.cuda.is_available():

                return 0.0

            props = torch.cuda.get_device_properties(0)

            return float(props.total_memory) / (1024**3)

        except Exception:  # noqa: BLE001

            return 0.0



    @staticmethod

    def _ollama_model_available(model_id: str) -> bool:

        """True when Ollama is reachable and the vision model tag (or family) is present."""

        try:

            request = urllib.request.Request(

                "http://127.0.0.1:11434/api/tags",

                method="GET",

            )

            with urllib.request.urlopen(request, timeout=2.0) as response:

                data = json.loads(response.read().decode("utf-8"))

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):

            return False

        names = {

            str(item.get("name", "")).split(":")[0].lower()

            for item in data.get("models", [])

            if isinstance(item, dict)

        }

        full_names = {

            str(item.get("name", "")).lower()

            for item in data.get("models", [])

            if isinstance(item, dict)

        }

        target = model_id.lower().strip()

        family = target.split(":")[0]

        return target in full_names or family in names or any(

            family in name for name in full_names

        )





def select_vision_adapter(config: VlmSelectionConfig) -> VlmChoice:

    return VlmSelector(config).select()


