"""Analyze YOLO detection crops / masks for colors, texture, and appearance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.clothing.clothing_analyzer import ClothingAnalysis, ClothingAnalyzer
from analysis.common.color_utils import (
    _mask_pixels,
    dominant_color_for_entity,
    secondary_color_name,
)
from core.contracts.detection import BoundingBox, Detection

_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_ANIMAL_LABELS = {
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
_FOLIAGE_COLORS = {
    "olive",
    "olive green",
    "forest green",
    "green",
    "dark green",
    "cyan",
}
_GROUND_BLEED = {"beige", "tan", "cream", "khaki", "blond", "olive", "brown"}


@dataclass(frozen=True)
class CropAnalysis:
    """Detailed visual analysis for one object crop/mask."""

    label: str
    dominant_color: str
    secondary_color: str
    material: str
    texture: str
    brightness: str
    edge_density: str
    clothing: ClothingAnalysis | None
    description: str
    confidence: float


class ObjectCropAnalyzer:
    """CPU-friendly crop/mask analyzer used after YOLO (+ SAM2 when available)."""

    def __init__(self) -> None:
        self._clothing = ClothingAnalyzer()

    def analyze(self, pixels: NDArray[np.uint8], detection: Detection) -> CropAnalysis:
        box = detection.bounding_box
        mask = detection.segmentation
        label = detection.label
        # Entity-bound color: inset + vegetation/ground rejection (not whole-scene color).
        dominant = dominant_color_for_entity(pixels, box, mask, label=label)
        secondary = secondary_color_name(pixels, box, mask, label=label)
        # Animals / bikes on grass — foliage leakage must not become object color.
        if label.lower() in _ANIMAL_LABELS and dominant in _FOLIAGE_COLORS:
            dominant = secondary if secondary not in _FOLIAGE_COLORS | {"unknown"} else "unknown"
        if label.lower() in _ANIMAL_LABELS and secondary in _FOLIAGE_COLORS:
            secondary = "unknown"
        if label.lower() in {"bicycle", "motorcycle", "bike", "skateboard"} and dominant in _FOLIAGE_COLORS:
            dominant = secondary if secondary not in _FOLIAGE_COLORS | {"unknown"} else "unknown"
        if label.lower() in {"sports ball", "ball", "frisbee"} and dominant in _GROUND_BLEED:
            dominant = secondary if secondary not in _GROUND_BLEED | {"unknown"} else "unknown"
        material = self._material(pixels, box, mask)
        texture = self._texture(pixels, box, mask)
        brightness = self._brightness_label(pixels, box, mask)
        edge_density = self._edge_density(pixels, box, mask)
        clothing: ClothingAnalysis | None = None
        if detection.label.lower() in _PERSON_LABELS:
            clothing = self._clothing.analyze(
                pixels,
                box,
                mask,
                detection_confidence=detection.confidence,
            )
        description = self._describe(detection.label, dominant, secondary, material, texture, clothing)
        conf = detection.confidence * (0.95 if mask is not None else 0.8)
        return CropAnalysis(
            label=detection.label,
            dominant_color=dominant,
            secondary_color=secondary,
            material=material,
            texture=texture,
            brightness=brightness,
            edge_density=edge_density,
            clothing=clothing,
            description=description,
            confidence=conf,
        )

    def analyze_all(
        self,
        pixels: NDArray[np.uint8],
        detections: tuple[Detection, ...],
    ) -> tuple[CropAnalysis, ...]:
        return tuple(self.analyze(pixels, detection) for detection in detections)

    def _material(self, pixels: NDArray[np.uint8], box: BoundingBox, mask: object) -> str:
        region = _mask_pixels(pixels, box, mask)  # type: ignore[arg-type]
        if region.size == 0:
            return "unknown"
        variance = float(region.astype(np.float32).std())
        if variance < 16:
            return "matte"
        if variance < 32:
            return "fabric"
        if variance < 48:
            return "textured"
        return "reflective"

    def _texture(self, pixels: NDArray[np.uint8], box: BoundingBox, mask: object) -> str:
        region = _mask_pixels(pixels, box, mask)  # type: ignore[arg-type]
        if region.size == 0:
            return "unknown"
        gray = region.mean(axis=2) if region.ndim == 3 else region.astype(np.float32)
        if gray.ndim == 1 or gray.shape[0] < 2:
            return "smooth"
        if gray.ndim == 2 and (gray.shape[0] < 2 or gray.shape[1] < 2):
            return "smooth"
        try:
            edges = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        except ValueError:
            return "smooth"
        if edges > 30:
            return "detailed"
        if edges > 14:
            return "moderate"
        return "smooth"

    def _brightness_label(self, pixels: NDArray[np.uint8], box: BoundingBox, mask: object) -> str:
        region = _mask_pixels(pixels, box, mask)  # type: ignore[arg-type]
        if region.size == 0:
            return "unknown"
        mean = float(region.astype(np.float32).mean())
        if mean < 70:
            return "dark"
        if mean > 180:
            return "bright"
        return "balanced"

    def _edge_density(self, pixels: NDArray[np.uint8], box: BoundingBox, mask: object) -> str:
        return self._texture(pixels, box, mask)

    def _describe(
        self,
        label: str,
        dominant: str,
        secondary: str,
        material: str,
        texture: str,
        clothing: ClothingAnalysis | None,
    ) -> str:
        if clothing is not None:
            parts = [f"{label} with {clothing.clothing_type} {clothing.clothing_style} look"]
            if clothing.shirt_color != "unknown":
                parts.append(f"{clothing.shirt_color} upper garment")
            if clothing.pants_color != "unknown":
                parts.append(f"{clothing.pants_color} lower garment")
            if clothing.hair_color != "unknown":
                parts.append(f"{clothing.hair_color} hair")
            if clothing.footwear_type != "unknown":
                parts.append(clothing.footwear_type)
            return ", ".join(parts)
        if secondary != dominant:
            return f"{dominant} {label} with {secondary} accents ({material}, {texture})"
        return f"{dominant} {label} ({material}, {texture})"
