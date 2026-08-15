"""Mask-aware clothing, footwear, accessory, and hair analysis with confidence gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.common.color_utils import (
    _mask_pixels,
    dominant_color_name,
    exclude_earth_ground_pixels,
    exclude_skin_pixels,
    exclude_vegetation_pixels,
    estimate_color,
    normalize_simple_color_name,
    region_color_name,
    secondary_color_name,
)
from core.contracts.detection import BoundingBox, SegmentationMask

_MIN_ANALYSIS_CONF = 0.55
_SKIP = frozenset({"unknown", "unlikely", "none detected", "not_applicable", "possible"})


@dataclass(frozen=True)
class ClothingAnalysis:
    """Structured clothing attributes for one detected person."""

    hair_color: str
    hair_length: str
    hairstyle: str
    shirt_color: str
    pants_color: str
    shoes_color: str
    clothing_color: str
    secondary_color: str
    clothing_type: str
    clothing_style: str
    clothing_texture: str
    sleeve_length: str
    jacket: str
    coat: str
    dress: str
    hoodie: str
    blazer: str
    sweater: str
    skirt: str
    jeans: str
    shorts: str
    footwear_type: str
    backpack: str
    handbag: str
    glasses: str
    sunglasses: str
    hat: str
    cap: str
    watch: str
    necklace: str
    earrings: str
    accessories: str
    dominant_colors: tuple[str, ...]
    confidence: float


class ClothingAnalyzer:
    """Heuristic clothing analysis; only high-confidence attributes are emitted."""

    def analyze(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None = None,
        *,
        detection_confidence: float = 0.7,
    ) -> ClothingAnalysis | None:
        if detection_confidence < _MIN_ANALYSIS_CONF:
            return None

        hair = self._map_hair_color(self._confident_band_color(pixels, box, "top", mask, exclude_skin=False))
        shirt = self._confident_band_color(pixels, box, "middle", mask, exclude_skin=True)
        pants = self._confident_band_color(pixels, box, "bottom", mask, exclude_skin=True)
        shoes = self._shoes_color(pixels, box, mask)
        clothing_type, type_conf = self._infer_clothing_type(box, shirt, pants, mask, pixels)
        # Keep soft garment cues; only wipe truly weak guesses.
        if type_conf < 0.56 or clothing_type in {"unknown", "casual"}:
            clothing_type = "unknown"
            type_conf = min(type_conf, 0.45)
        style = self._clothing_style(clothing_type) if clothing_type != "unknown" else "unknown"
        # Hair color is often contaminated — keep only high-confidence hair estimates.
        if hair != "unknown":
            hair_box = BoundingBox(
                box.x_min + box.width * 0.2,
                box.y_min,
                box.x_max - box.width * 0.2,
                box.y_min + max(1.0, box.height * 0.18),
            )
            hair_est = estimate_color(_mask_pixels(pixels, hair_box, mask))
            hair = hair if hair_est.confidence >= 0.68 and hair_est.name == hair else "unknown"
        texture = self._texture_label(pixels, box, mask)
        sleeve = self._sleeve_length(pixels, box, mask)
        footwear = self._footwear_type(pixels, box, shoes, mask)

        hat, cap = self._headwear(pixels, box, hair, mask)
        # Never promote garments from color alone — unknown beats hallucination.
        hoodie = "likely" if clothing_type in {"hoodie", "hooded sweatshirt"} and type_conf >= 0.65 else "unlikely"
        dress = "likely" if clothing_type in {"dress", "traditional clothing"} and type_conf >= 0.62 else "unlikely"
        jacket = "likely" if clothing_type in {"jacket", "coat", "windbreaker"} and type_conf >= 0.65 else "unlikely"
        coat = "likely" if clothing_type == "coat" and type_conf >= 0.65 else "unlikely"
        blazer = (
            "likely"
            if clothing_type in {"formal suit", "blazer"}
            and type_conf >= 0.65
            and shirt in {"white", "cream", "light gray", "light blue"}
            else "unlikely"
        )
        sweater = "likely" if clothing_type in {"sweater", "cardigan"} and type_conf >= 0.65 else "unlikely"
        skirt = "likely" if clothing_type == "skirt" and type_conf >= 0.60 else "unlikely"
        jeans = (
            "likely"
            if (
                (clothing_type == "jeans" and type_conf >= 0.60)
                or (
                    pants in {"navy blue", "blue", "sky blue", "royal blue"}
                    and clothing_type
                    in {"unknown", "t-shirt", "hoodie", "shirt", "polo", "jacket", "sweater", "sportswear"}
                    and pants != shirt
                )
            )
            else "unlikely"
        )
        shorts = "likely" if clothing_type == "shorts" and type_conf >= 0.60 else "unlikely"
        # Never invent backpacks from clothing silhouette / side bulk — that hallucinated
        # accessories into farm/office captions. Only YOLO detections may introduce them.
        backpack = "unlikely"
        handbag = "unlikely"
        # Eyewear is not inferred from silhouette — requires strong detector evidence.
        glasses = "unlikely"
        sunglasses = "unlikely"
        watch = "unknown"
        necklace = "unknown"
        earrings = "unknown"
        accessories = self._accessories(hat, cap, backpack)
        hair_length, hairstyle = self._hair_attributes(box, hair)

        # Multi-source clothing confidence (detection + mask + type + color support).
        color_support = sum(1 for c in (shirt, pants, shoes) if c != "unknown") / 3.0
        conf = (
            detection_confidence * 0.28
            + (0.88 if mask is not None else 0.45) * 0.22
            + type_conf * 0.20
            + color_support * 0.15
            + 0.70 * 0.10
            + 0.58 * 0.05
        )
        if shirt == "unknown" and pants == "unknown" and hair == "unknown":
            conf -= 0.12
        conf = max(0.0, min(0.95, conf))
        if conf < _MIN_ANALYSIS_CONF:
            return None

        secondary = secondary_color_name(pixels, box, mask)
        # Never fall back to whole-person dominant color (background bleed).
        clothing_color = shirt if shirt != "unknown" else pants

        return ClothingAnalysis(
            hair_color=hair,
            hair_length=hair_length,
            hairstyle=hairstyle,
            shirt_color=shirt,
            pants_color=pants,
            shoes_color=shoes,
            clothing_color=clothing_color,
            secondary_color=secondary,
            clothing_type=clothing_type,
            clothing_style=style,
            clothing_texture=texture,
            sleeve_length=sleeve,
            jacket=jacket,
            coat=coat,
            dress=dress,
            hoodie=hoodie,
            blazer=blazer,
            sweater=sweater,
            skirt=skirt,
            jeans=jeans,
            shorts=shorts,
            footwear_type=footwear,
            backpack=backpack,
            handbag=handbag,
            glasses=glasses,
            sunglasses=sunglasses,
            hat=hat,
            cap=cap,
            watch=watch,
            necklace=necklace,
            earrings=earrings,
            accessories=accessories,
            dominant_colors=self._palette(pixels, box, mask, hair, shirt, pants),
            confidence=conf,
        )

    def _confident_band_color(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        band: str,
        mask: SegmentationMask | None,
        *,
        exclude_skin: bool = False,
    ) -> str:
        name = region_color_name(
            pixels,
            box,
            vertical_band=band,
            mask=mask,
            exclude_skin=exclude_skin,
        )
        return name if name not in _SKIP else "unknown"

    def _map_hair_color(self, name: str) -> str:
        if name in {"yellow", "cream", "beige", "mustard"}:
            return "blond"
        if name == "charcoal":
            return "black"
        return name

    def _shoes_color(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None,
    ) -> str:
        # Footwear band only — not the full lower body or surrounding ground.
        shoe_box = BoundingBox(
            box.x_min + box.width * 0.18,
            box.y_max - max(1.0, box.height * 0.12),
            box.x_max - box.width * 0.18,
            box.y_max - box.height * 0.01,
        )
        region = _mask_pixels(pixels, shoe_box, mask, inset_x=0.12, inset_y=0.08)
        region = exclude_skin_pixels(region)
        region = exclude_vegetation_pixels(region)
        region = exclude_earth_ground_pixels(region)
        estimate = estimate_color(region)
        if estimate.confidence < 0.58 or estimate.name == "unknown":
            return "unknown"
        name = normalize_simple_color_name(estimate.name)
        if name in {"olive", "green", "beige", "tan", "khaki", "cream"} and estimate.confidence < 0.78:
            return "unknown"
        return name if name not in _SKIP else "unknown"

    def _infer_clothing_type(
        self,
        box: BoundingBox,
        shirt: str,
        pants: str,
        mask: SegmentationMask | None,
        pixels: NDArray[np.uint8],
    ) -> tuple[str, float]:
        ratio = box.height / max(box.width, 1.0)
        mid = _mask_pixels(
            pixels,
            BoundingBox(box.x_min, box.y_min + box.height * 0.3, box.x_max, box.y_min + box.height * 0.65),
            mask,
        )
        brightness = float(mid.astype(np.float32).mean()) if mid.size else 128.0
        variance = float(mid.astype(np.float32).std()) if mid.size else 0.0
        sleeve = self._sleeve_length(pixels, box, mask)

        # High edge energy in torso → athletic fabric more likely than formal wear.
        athletic_cue = variance > 42 and brightness > 100

        if ratio >= 2.3 and shirt == pants and shirt != "unknown" and variance < 30:
            return "dress", 0.8
        # Formal wear first (strict) so dark suits are not mislabeled as hoodies.
        if (
            sleeve == "long"
            and variance < 20
            and brightness < 120
            and not athletic_cue
            and shirt in {"black", "charcoal", "white", "cream", "light gray"}
            and pants in {"black", "charcoal", "dark gray", "navy blue"}
            and shirt != pants
        ):
            if shirt in {"white", "cream", "light gray"}:
                return "blazer", 0.72
            return "formal suit", 0.76
        if (
            sleeve == "long"
            and variance < 18
            and brightness < 110
            and not athletic_cue
            and shirt == pants
            and shirt in {"black", "charcoal"}
        ):
            return "formal suit", 0.7
        texture = self._texture_label(pixels, box, mask)
        soft_knit = texture in {"soft_knit", "smooth"}
        woven = texture in {"woven", "textured", "striped"}
        # Layering cue: torso darker than lower body often means outerwear over a base layer.
        lower = _mask_pixels(
            pixels,
            BoundingBox(box.x_min, box.y_min + box.height * 0.55, box.x_max, box.y_min + box.height * 0.85),
            mask,
        )
        lower_brightness = float(lower.astype(np.float32).mean()) if lower.size else brightness
        outerwear_layer = brightness + 12.0 < lower_brightness and sleeve == "long"

        # Jacket / coat / windbreaker before hoodie/shirt — outerwear must never become shirt.
        if sleeve == "long" and shirt != "unknown" and not athletic_cue:
            if outerwear_layer or woven or variance >= 30:
                if brightness < 95 and (variance >= 50 or outerwear_layer):
                    return "coat", 0.74
                if variance >= 46 and brightness > 118 and woven:
                    return "windbreaker", 0.66
                if outerwear_layer or variance >= 30 or woven:
                    return "jacket", 0.72

        # Hoodie / hooded sweatshirt: soft knit torso — never when outerwear layering is present.
        hoodie_colors = {
            "charcoal", "dark gray", "black", "navy blue", "navy", "burgundy", "maroon",
            "olive green", "olive", "red", "green", "blue", "royal blue", "purple", "pink",
            "beige", "brown", "forest green", "sky blue", "light blue",
        }
        if (
            sleeve == "long"
            and 16 <= variance <= 48
            and soft_knit
            and not outerwear_layer
            and not woven
            and shirt in hoodie_colors
            and not athletic_cue
            and brightness < 155
        ):
            return "hoodie", 0.72
        # Pants cues must not replace an upper-garment type — jeans/cargo are secondary flags.
        if athletic_cue and shirt in {
            "red", "yellow", "mustard", "green", "orange", "sky blue", "white", "navy blue", "royal blue",
        }:
            if variance > 50:
                return "jersey", 0.68
            return "sportswear", 0.7
        # Sweater / cardigan: soft knit + warm neutrals; do not steal hoodies or outerwear.
        if (
            sleeve == "long"
            and variance < 24
            and soft_knit
            and not woven
            and shirt in {"beige", "cream", "brown", "olive green", "olive", "tan", "khaki"}
            and not athletic_cue
        ):
            if outerwear_layer:
                return "cardigan", 0.68
            return "sweater", 0.70
        if sleeve == "short" and shirt not in {"unknown"} and pants != "unknown" and not outerwear_layer:
            if variance < 35 and shirt in {"white", "sky blue", "navy blue", "navy", "red", "light blue"}:
                return "polo", 0.66
            return "t-shirt", 0.70
        if (
            sleeve == "long"
            and shirt in {"white", "sky blue", "cream", "light blue"}
            and pants != "unknown"
            and variance < 30
            and not woven
            and not outerwear_layer
            and soft_knit
        ):
            return "shirt", 0.66
        # Uniform: matching torso+lower with institutional colors.
        if (
            shirt != "unknown"
            and pants != "unknown"
            and shirt == pants
            and sleeve != "unknown"
            and not athletic_cue
            and not outerwear_layer
            and shirt
            in {
                "navy blue",
                "navy",
                "dark green",
                "forest green",
                "black",
                "charcoal",
                "sky blue",
                "light blue",
            }
            and variance < 36
        ):
            return "uniform", 0.66
        if shirt != "unknown" and pants == "unknown" and ratio >= 2.15 and variance < 35 and not outerwear_layer:
            return "dress", 0.64
        # Only when no upper garment is evidenced, report pants-led types.
        if shirt == "unknown" and pants in {"navy blue", "blue", "sky blue", "royal blue"}:
            return "jeans", 0.60
        if shirt == "unknown" and pants in {"beige", "olive green", "olive", "brown", "khaki", "tan"} and 28 < variance < 58:
            return "cargo pants", 0.60
        if ratio <= 1.5 and pants != "unknown" and brightness > 115 and sleeve != "long" and shirt == "unknown":
            return "shorts", 0.62
        return "unknown", 0.35

    def _clothing_style(self, clothing_type: str) -> str:
        mapping = {
            "formal suit": "formal",
            "blazer": "formal",
            "sportswear": "athletic",
            "jersey": "athletic",
            "hoodie": "streetwear",
            "hooded sweatshirt": "streetwear",
            "jeans": "casual",
            "shorts": "casual",
            "dress": "formal_or_smart",
            "sweater": "smart_casual",
            "cardigan": "smart_casual",
            "long sleeve shirt": "casual",
            "shirt": "casual",
            "t-shirt": "casual",
            "polo": "casual",
            "polo shirt": "casual",
            "coat": "outerwear",
            "jacket": "outerwear",
            "windbreaker": "outerwear",
            "skirt": "smart_casual",
            "jeans": "casual",
            "cargo pants": "casual",
            "traditional clothing": "traditional",
            "uniform": "uniform",
        }
        return mapping.get(clothing_type, "casual")

    def _texture_label(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None,
    ) -> str:
        region = _mask_pixels(pixels, box, mask)
        if region.size < 32:
            return "unknown"
        gray = region.astype(np.float32).mean(axis=2) if region.ndim == 3 else region.astype(np.float32)
        std = float(gray.std())
        # Stripe heuristic: strong horizontal/vertical periodicity in mid band.
        if gray.ndim == 2 and gray.shape[0] >= 12 and gray.shape[1] >= 12:
            row_var = float(np.var(gray.mean(axis=1)))
            col_var = float(np.var(gray.mean(axis=0)))
            if max(row_var, col_var) > 120 and min(row_var, col_var) < 40:
                return "striped"
        if std < 18:
            return "smooth"
        if std < 40:
            return "soft_knit"
        if std < 65:
            return "woven"
        return "textured"

    def _sleeve_length(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None,
    ) -> str:
        arm = BoundingBox(
            box.x_min,
            box.y_min + box.height * 0.3,
            box.x_min + box.width * 0.25,
            box.y_min + box.height * 0.6,
        )
        region = _mask_pixels(pixels, arm, mask)
        if region.size == 0:
            return "unknown"
        estimate = estimate_color(region)
        # Weak arm-band color → omit sleeve rather than invent long/short.
        if estimate.name == "unknown" or estimate.confidence < 0.58:
            return "unknown"
        brightness = float(region.astype(np.float32).mean())
        return "long" if brightness < 135 else "short"

    def _footwear_type(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        shoes_color: str,
        mask: SegmentationMask | None,
    ) -> str:
        if shoes_color == "unknown":
            return "unknown"
        shoe_box = BoundingBox(box.x_min, box.y_max - max(1.0, box.height * 0.12), box.x_max, box.y_max)
        region = _mask_pixels(pixels, shoe_box, mask)
        if region.size == 0:
            return "unknown"
        brightness = float(region.astype(np.float32).mean())
        std = float(region.astype(np.float32).std())
        # Footwear is easy to get wrong — require strong cues or omit.
        if shoes_color in {"white", "cream", "red", "sky blue"} and std > 30 and brightness > 100:
            return "sneakers"
        if shoes_color in {"brown", "beige"} and brightness < 100 and std < 35 and box.height / max(box.width, 1.0) > 2.0:
            return "boots"
        return "unknown"

    def _headwear(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        hair: str,
        mask: SegmentationMask | None,
    ) -> tuple[str, str]:
        top = BoundingBox(box.x_min, box.y_min, box.x_max, box.y_min + max(1.0, box.height * 0.12))
        region = _mask_pixels(pixels, top, mask)
        if region.size == 0:
            return "unlikely", "unlikely"
        estimate = estimate_color(region)
        brightness = float(region.astype(np.float32).mean())
        if estimate.name != "unknown" and estimate.name != hair and brightness < 70:
            return "likely", "unlikely"
        if estimate.name in {"red", "navy blue", "sky blue", "green"} and estimate.name != hair:
            return "unlikely", "likely"
        return "unlikely", "unlikely"

    def _side_bulk(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None,
    ) -> bool:
        """Legacy silhouette probe — kept for API compatibility; unused for accessories."""
        side = BoundingBox(
            box.x_min,
            box.y_min + box.height * 0.25,
            box.x_min + box.width * 0.2,
            box.y_min + box.height * 0.7,
        )
        region = _mask_pixels(pixels, side, mask)
        return bool(region.size and float(region.astype(np.float32).std()) > 48)

    def _accessories(self, hat: str, cap: str, backpack: str) -> str:
        items = []
        if hat == "likely":
            items.append("hat")
        if cap == "likely":
            items.append("cap")
        # backpack intentionally ignored here — requires a high-confidence YOLO detection.
        _ = backpack
        return ", ".join(items) if items else "none detected"

    def _hair_attributes(self, box: BoundingBox, hair: str) -> tuple[str, str]:
        if hair == "unknown":
            return "unknown", "unknown"
        ratio = box.height / max(box.width, 1.0)
        if ratio >= 2.0:
            return "medium_or_long", "loose"
        if hair in {"blond", "yellow", "cream", "white"}:
            return "short_or_medium", "visible"
        return "short", "short_or_covered"

    def _palette(
        self,
        pixels: NDArray[np.uint8],
        box: BoundingBox,
        mask: SegmentationMask | None,
        hair: str,
        shirt: str,
        pants: str,
    ) -> tuple[str, ...]:
        primary = dominant_color_name(pixels, box, mask)
        secondary = secondary_color_name(pixels, box, mask)
        return tuple(
            dict.fromkeys(
                item for item in (primary, secondary, hair, shirt, pants) if item not in _SKIP and item
            )
        )
