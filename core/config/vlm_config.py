"""Vision-language model catalog and selection configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VlmModelIds:
    """Configurable model identifiers — never hardcode outside config."""

    gemma_vision: str
    florence_base: str
    florence_plain: str
    florence_large: str
    moondream: str
    blip2: str
    blip: str
    qwen: str
    internvl: str


@dataclass(frozen=True)
class VlmSelectionConfig:
    """Automatic VLM selection policy for 2GB-class hardware."""

    auto_select: bool
    preferred_adapter: str
    model_ids: VlmModelIds
    min_vram_gemma_vision_gb: float
    min_vram_florence_base_gb: float
    min_vram_florence_plain_gb: float
    min_vram_moondream_gb: float
    min_vram_blip2_gb: float
    min_vram_florence_large_gb: float
    min_vram_qwen_gb: float
    min_vram_internvl_gb: float
