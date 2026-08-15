"""Factory for vision-language adapters from selector choices."""

from __future__ import annotations

from language.vlm.adapters.base import BaseVisionAdapter
from language.vlm.adapters.blip2_adapter import Blip2VisionAdapter
from language.vlm.adapters.blip_adapter import BlipVisionAdapter
from language.vlm.adapters.florence_adapter import FlorenceVisionAdapter
from language.vlm.adapters.internvl_adapter import InternVLVisionAdapter
from language.vlm.adapters.moondream_adapter import MoondreamVisionAdapter
from language.vlm.adapters.ollama_vision_adapter import OllamaVisionAdapter
from language.vlm.adapters.qwen_adapter import QwenVisionAdapter
from language.vlm.selector import VlmChoice


def create_vision_model(choice: VlmChoice, preferred_device: str) -> BaseVisionAdapter:
    """Create an adapter instance for the selected choice."""
    name = choice.adapter_name
    if name == "gemma_vision":
        return OllamaVisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "blip":
        return BlipVisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "blip2":
        return Blip2VisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "moondream":
        return MoondreamVisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "qwen":
        return QwenVisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "internvl":
        return InternVLVisionAdapter(choice.model_id, preferred_device=preferred_device)
    if name == "florence_large":
        return FlorenceVisionAdapter(choice.model_id, "florence_large", preferred_device=preferred_device)
    if name == "florence_plain":
        return FlorenceVisionAdapter(choice.model_id, "florence_plain", preferred_device=preferred_device)
    return FlorenceVisionAdapter(choice.model_id, "florence_base", preferred_device=preferred_device)
