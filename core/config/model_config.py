"""AI model configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path

from core.config.vlm_config import VlmSelectionConfig


@dataclass(frozen=True)
class YoloModelConfig:
    """YOLO detector configuration."""

    variant: str
    weights_path: Path | None
    confidence_threshold: float
    iou_threshold: float
    preferred_device: str


@dataclass(frozen=True)
class BlipModelConfig:
    """BLIP vision-language model configuration."""

    model_id: str
    preferred_device: str
    max_length: int


@dataclass(frozen=True)
class FlorenceModelConfig:
    """Florence-2 detailed vision-language configuration."""

    model_id: str
    preferred_device: str
    max_new_tokens: int
    fallback_to_blip: bool


@dataclass(frozen=True)
class GemmaModelConfig:
    """Gemma reasoning model configuration."""

    model_id: str
    preferred_device: str
    quantization: str
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True)
class PluginConfig:
    """Plugin identifier bindings for swappable engines."""

    detection_plugin: str
    vision_language_plugin: str
    reasoning_plugin: str


@dataclass(frozen=True)
class ModelConfig:
    """Aggregate model configuration."""

    yolo: YoloModelConfig
    blip: BlipModelConfig
    florence: FlorenceModelConfig
    gemma: GemmaModelConfig
    plugins: PluginConfig
    vlm: VlmSelectionConfig
