"""Hardware-aware model recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from core.constants.model_kinds import ModelKind
from model_management.catalog import PRODUCTION_MODELS, ProductionModelSpec


@dataclass(frozen=True)
class HardwareProfile:
    """Detected host hardware capabilities."""

    ram_total_gb: float
    ram_available_gb: float
    vram_total_mb: float
    cuda_available: bool
    gpu_name: str


@dataclass(frozen=True)
class ModelResourceAdvice:
    """Recommendation for one model on current hardware."""

    spec: ProductionModelSpec
    recommended: bool
    warning: str


@dataclass(frozen=True)
class HardwareAssessment:
    """Aggregate hardware suitability report."""

    profile: HardwareProfile
    advice: tuple[ModelResourceAdvice, ...]

    @property
    def has_warnings(self) -> bool:
        return any(item.warning for item in self.advice)


def detect_hardware() -> HardwareProfile:
    """Probe RAM and GPU availability."""
    mem = psutil.virtual_memory()
    vram_mb = 0.0
    cuda = False
    gpu_name = "None"
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            props = torch.cuda.get_device_properties(0)
            vram_mb = float(props.total_memory) / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return HardwareProfile(
        ram_total_gb=mem.total / (1024**3),
        ram_available_gb=mem.available / (1024**3),
        vram_total_mb=vram_mb,
        cuda_available=cuda,
        gpu_name=gpu_name,
    )


def assess_models(profile: HardwareProfile | None = None) -> HardwareAssessment:
    """Recommend models suitable for ~2 GB VRAM / 8–16 GB RAM targets."""
    hardware = profile or detect_hardware()
    advice: list[ModelResourceAdvice] = []
    for spec in PRODUCTION_MODELS:
        warning = ""
        recommended = True
        if spec.kind == ModelKind.YOLO and hardware.vram_total_mb < 1024 and hardware.cuda_available:
            warning = "YOLO11x may be slow on GPUs under 1 GB VRAM; CPU fallback available."
        if spec.kind == ModelKind.BLIP and hardware.ram_total_gb < 8:
            recommended = False
            warning = "BLIP Large requires at least 8 GB RAM for stable operation."
        if spec.kind == ModelKind.GEMMA:
            if hardware.ram_total_gb < 8:
                recommended = False
                warning = "Gemma 2 2B requires at least 8 GB RAM."
            elif hardware.vram_total_mb < 2048 and hardware.cuda_available:
                warning = "INT4 quantization recommended for GPUs under 2 GB VRAM."
        advice.append(ModelResourceAdvice(spec=spec, recommended=recommended, warning=warning))
    return HardwareAssessment(profile=hardware, advice=tuple(advice))
