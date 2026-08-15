"""Startup-time model discovery and validation."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from core.utils.paths import normalize_optional_path
from services.runtime.yolo_weights import resolve_yolo_weights_path


@dataclass(frozen=True)
class ModelDiscoveryEntry:
    """Availability status for one model kind."""

    kind: str
    identifier: str
    available: bool
    detail: str


@dataclass(frozen=True)
class ModelDiscoveryReport:
    """Aggregated model discovery results."""

    entries: tuple[ModelDiscoveryEntry, ...]
    scanned_files: tuple[str, ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            f"{entry.kind}: {entry.detail}"
            for entry in self.entries
            if not entry.available
        )


def discover_models(model_config: ModelConfig, models_dir: Path) -> ModelDiscoveryReport:
    """Discover configured models and scan the models directory."""
    models_dir.mkdir(parents=True, exist_ok=True)
    scanned = tuple(sorted(str(path.name) for path in models_dir.iterdir() if path.is_file()))

    entries: list[ModelDiscoveryEntry] = []

    yolo_weights = resolve_yolo_weights_path(
        variant=model_config.yolo.variant,
        configured_path=normalize_optional_path(model_config.yolo.weights_path),
        search_paths=(models_dir.resolve(),),
    )
    if yolo_weights is not None:
        yolo_detail = f"Weights found at {yolo_weights}"
        yolo_available = True
    elif normalize_optional_path(model_config.yolo.weights_path) is not None:
        configured = normalize_optional_path(model_config.yolo.weights_path)
        yolo_detail = (
            f"Weights not found at {configured}; "
            f"variant {model_config.yolo.variant} may download on first use"
        )
        yolo_available = False
    else:
        yolo_detail = f"Using variant {model_config.yolo.variant}; weights may download on first use"
        yolo_available = any(name.endswith(".pt") for name in scanned)
    entries.append(
        ModelDiscoveryEntry(
            kind=ModelKind.YOLO.name.lower(),
            identifier=model_config.yolo.variant,
            available=yolo_available,
            detail=yolo_detail,
        )
    )

    entries.append(
        ModelDiscoveryEntry(
            kind=ModelKind.BLIP.name.lower(),
            identifier=model_config.blip.model_id,
            available=bool(model_config.blip.model_id.strip()),
            detail="Configured for Hugging Face download on first use"
            if model_config.blip.model_id.strip()
            else "Model ID missing",
        )
    )
    entries.append(
        ModelDiscoveryEntry(
            kind="florence2",
            identifier=model_config.florence.model_id,
            available=bool(model_config.florence.model_id.strip()),
            detail=(
                f"Florence-2 configured ({model_config.florence.model_id}); "
                f"BLIP fallback={'on' if model_config.florence.fallback_to_blip else 'off'}"
            )
            if model_config.florence.model_id.strip()
            else "Florence-2 model ID missing",
        )
    )
    entries.append(
        ModelDiscoveryEntry(
            kind=ModelKind.GEMMA.name.lower(),
            identifier=model_config.gemma.model_id,
            available=bool(model_config.gemma.model_id.strip()),
            detail="Configured for Hugging Face download on first use"
            if model_config.gemma.model_id.strip()
            else "Model ID missing",
        )
    )

    sam2_available = importlib.util.find_spec("sam2") is not None or any(
        "sam" in name.lower() for name in scanned
    )
    entries.append(
        ModelDiscoveryEntry(
            kind="sam2",
            identifier="sam2",
            available=sam2_available,
            detail="SAM2 package or weights found in models/"
            if sam2_available
            else "SAM2 optional; segmentation refinement falls back when unavailable",
        )
    )

    realesrgan_candidates = tuple(
        path for path in models_dir.rglob("*") if path.is_file() and "realesrgan" in path.name.lower()
    )
    entries.append(
        ModelDiscoveryEntry(
            kind="realesrgan",
            identifier="RealESRGAN",
            available=bool(realesrgan_candidates),
            detail=f"Found {len(realesrgan_candidates)} RealESRGAN file(s) under models/"
            if realesrgan_candidates
            else "RealESRGAN optional; super-resolution uses OpenCV fallback",
        )
    )

    ollama_available = shutil.which("ollama") is not None
    entries.append(
        ModelDiscoveryEntry(
            kind="ollama",
            identifier=model_config.gemma.model_id or "gemma2:2b",
            available=ollama_available,
            detail="Ollama CLI available for local Gemma inference"
            if ollama_available
            else "Ollama not installed; Gemma uses Hugging Face provider",
        )
    )

    return ModelDiscoveryReport(entries=tuple(entries), scanned_files=scanned)
