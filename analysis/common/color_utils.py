"""Competition-grade color analysis with LAB/HSV, illumination normalization, and confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.contracts.detection import BoundingBox, SegmentationMask

# Adaptive base gate — uniform regions may pass slightly lower; mixed stay strict.
_MIN_COLOR_CONFIDENCE = 0.55

# Named palette calibrated against OpenCV RGB→LAB for natural language colors.
# Gray entries exist only for truly achromatic samples — never as a default guess.
_NAMED_LAB: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("black", (2.4, 0.0, 0.0)),
    ("charcoal", (16.1, 1.0, -1.0)),
    ("dark gray", (34.1, 0.0, 0.0)),
    ("light gray", (73.3, 0.0, 0.0)),
    ("white", (96.5, 0.0, 0.0)),
    ("cream", (93.3, -1.0, 13.0)),
    ("beige", (77.6, 1.0, 23.0)),
    ("tan", (62.0, 8.0, 28.0)),
    ("khaki", (68.5, -2.0, 36.0)),
    ("brown", (33.7, 14.0, 25.0)),
    ("maroon", (22.7, 33.0, 11.0)),
    ("burgundy", (19.6, 33.0, 6.0)),
    ("red", (44.3, 61.0, 41.0)),
    ("orange", (64.3, 32.0, 61.0)),
    ("mustard", (67.8, 4.0, 63.0)),
    ("yellow", (87.1, -10.0, 78.0)),
    ("blond", (74.5, 3.0, 35.0)),
    ("olive", (48.0, -10.0, 28.0)),
    ("olive green", (44.3, -13.0, 32.0)),
    ("dark green", (24.0, -18.0, 14.0)),
    ("forest green", (29.8, -27.0, 19.0)),
    ("green", (58.4, -50.0, 38.0)),
    ("cyan", (71.0, -31.0, -15.0)),
    ("sky blue", (71.0, -6.0, -31.0)),
    ("light blue", (78.0, -8.0, -18.0)),
    ("royal blue", (28.0, 28.0, -55.0)),
    ("navy blue", (15.7, 16.0, -35.0)),
    ("navy", (14.0, 12.0, -30.0)),
    ("blue", (31.4, 35.0, -66.0)),
    ("purple", (34.1, 47.0, -49.0)),
    ("pink", (68.2, 38.0, -1.0)),
)
_GRAY_NAMES = frozenset({"dark gray", "light gray"})
_STRICT_NEUTRALS = frozenset({"black", "charcoal", "white", "cream", "beige"})
_COLOR_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"olive", "olive green", "dark green", "forest green", "green"}),
    frozenset({"navy", "navy blue", "royal blue", "blue", "sky blue", "light blue", "cyan"}),
    frozenset({"burgundy", "maroon", "red", "pink"}),
    frozenset({"beige", "cream", "tan", "khaki", "blond", "white"}),
    frozenset({"black", "charcoal"}),
    frozenset({"brown", "tan", "khaki", "mustard"}),
)


def _same_color_family(name_a: str, name_b: str) -> bool:
    for family in _COLOR_FAMILIES:
        if name_a in family and name_b in family:
            return True
    return False


@dataclass(frozen=True)
class ColorEstimate:
    """Color name with confidence; omit when confidence is low."""

    name: str
    confidence: float
    lab: tuple[float, float, float]


def _region_pixels(pixels: NDArray[np.uint8], box: BoundingBox) -> NDArray[np.uint8]:
    height, width = pixels.shape[:2]
    x_min = int(max(0, min(width - 1, box.x_min)))
    y_min = int(max(0, min(height - 1, box.y_min)))
    x_max = int(max(x_min + 1, min(width, box.x_max)))
    y_max = int(max(y_min + 1, min(height, box.y_max)))
    return pixels[y_min:y_max, x_min:x_max]


def _inset_box(box: BoundingBox, *, x_frac: float = 0.12, y_frac: float = 0.08) -> BoundingBox:
    """Shrink box inward to reduce background contamination."""
    dx = box.width * x_frac
    dy = box.height * y_frac
    return BoundingBox(
        box.x_min + dx,
        box.y_min + dy,
        max(box.x_min + dx + 1.0, box.x_max - dx),
        max(box.y_min + dy + 1.0, box.y_max - dy),
    )


def _mask_pixels(
    pixels: NDArray[np.uint8],
    box: BoundingBox,
    mask: SegmentationMask | None,
    *,
    inset_x: float | None = None,
    inset_y: float | None = None,
) -> NDArray[np.uint8]:
    """Sample pixels using SAM2 polygon mask when available, else inset bbox crop."""
    if mask is not None:
        sample_box = box
    elif inset_x is not None:
        sample_box = _inset_box(box, x_frac=inset_x, y_frac=inset_y if inset_y is not None else inset_x)
    else:
        sample_box = _inset_box(box)
    region = _region_pixels(pixels, sample_box)
    if mask is None or not mask.polygon or region.size == 0:
        return region
    try:
        import cv2
    except ImportError:
        return region
    height, width = pixels.shape[:2]
    canvas = np.zeros((height, width), dtype=np.uint8)
    points = np.array(mask.polygon, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 3:
        return region
    cv2.fillPoly(canvas, [points], 255)  # type: ignore[call-overload]
    x_min = int(max(0, min(width - 1, sample_box.x_min)))
    y_min = int(max(0, min(height - 1, sample_box.y_min)))
    x_max = int(max(x_min + 1, min(width, sample_box.x_max)))
    y_max = int(max(y_min + 1, min(height, sample_box.y_max)))
    local_mask = canvas[y_min:y_max, x_min:x_max]
    if local_mask.shape[:2] != region.shape[:2] or int(local_mask.sum()) == 0:
        return region
    selected = region[local_mask > 0]
    if selected.size == 0:
        return region
    return selected.reshape(-1, 1, region.shape[2]) if selected.ndim == 2 else selected


def exclude_skin_pixels(region: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Drop skin-like pixels so clothing color is not contaminated by arms/face."""
    if region.size == 0 or region.ndim < 3:
        return region
    flat = region.reshape(-1, region.shape[-1])[:, :3].astype(np.float32)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    # Compact YCbCr-ish skin rule in RGB.
    skin = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )
    keep = flat[~skin]
    if keep.shape[0] < max(8, int(0.15 * flat.shape[0])):
        return region
    return keep.reshape(-1, 1, 3).astype(np.uint8)


def exclude_vegetation_pixels(region: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Drop grass/foliage-like greens that bleed into vehicle/animal/ball crops."""
    if region.size == 0 or region.ndim < 3:
        return region
    flat = region.reshape(-1, region.shape[-1])[:, :3].astype(np.float32)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    foliage = (
        (g > r + 8)
        & (g > b + 5)
        & (g > 35)
        & (g < 220)
        & ((g - np.minimum(r, b)) > 12)
    )
    keep = flat[~foliage]
    if keep.shape[0] < max(8, int(0.18 * flat.shape[0])):
        return region
    return keep.reshape(-1, 1, 3).astype(np.uint8)


def exclude_earth_ground_pixels(region: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Drop beige/tan dirt/ground pixels that contaminate sports-ball crops."""
    if region.size == 0 or region.ndim < 3:
        return region
    flat = region.reshape(-1, region.shape[-1])[:, :3].astype(np.float32)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    mean = (r + g + b) / 3.0
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    # Warm midtones with modest chroma → dirt/sand, not a painted ball.
    earth = (
        (mean > 70)
        & (mean < 215)
        & (r >= g - 5)
        & (g >= b - 8)
        & (r > b + 6)
        & (spread < 75)
        & (spread > 6)
    )
    keep = flat[~earth]
    if keep.shape[0] < max(6, int(0.12 * flat.shape[0])):
        return region
    return keep.reshape(-1, 1, 3).astype(np.uint8)


_SIMPLE_COLOR_MAP = {
    # Collapse only aliases that are not in the stable competition vocabulary.
    "cyan": "light blue",
    "sky blue": "light blue",
    "navy blue": "dark blue",
    "navy": "dark blue",
    "royal blue": "blue",
    "olive green": "olive",
    "forest green": "green",
    "dark green": "green",
    "dark gray": "gray",
    "light gray": "gray",
    "charcoal": "black",
    "mustard": "yellow",
    "blond": "beige",
    "khaki": "beige",
    "tan": "beige",
    "maroon": "burgundy",
    # Keep cream / olive / burgundy / dark blue as-is (do not flatten them away).
}


def normalize_simple_color_name(name: str) -> str:
    """Map internal palette names onto a compact human-readable vocabulary."""
    raw = (name or "").strip().lower()
    if not raw or raw == "unknown":
        return "unknown"
    return _SIMPLE_COLOR_MAP.get(raw, raw)


_GROUND_TONES = frozenset({"beige", "tan", "cream", "khaki", "blond", "olive", "brown"})
_FOLIAGE_TONES = frozenset(
    {"olive", "olive green", "forest green", "green", "dark green", "cyan"}
)
_SPORTS_SMALL = frozenset(
    {"sports ball", "ball", "frisbee", "apple", "orange", "clock", "baseball"}
)
_VEHICLE_LABELS = frozenset(
    {"bicycle", "motorcycle", "bike", "skateboard", "car", "bus", "truck", "train"}
)
_ANIMAL_LABELS_COLOR = frozenset(
    {"dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "zebra", "giraffe"}
)
# Large appliances/fixtures: bbox often includes wood cabinets/walls → warm bleed.
_APPLIANCE_LABELS = frozenset(
    {
        "refrigerator",
        "oven",
        "microwave",
        "dishwasher",
        "toaster",
        "sink",
        "tv",
        "laptop",
        "monitor",
        "washing machine",
    }
)
_CABINET_WOOD_TONES = frozenset({"beige", "tan", "cream", "khaki", "blond", "brown", "olive"})


def _high_luminance_neutral_name(region: NDArray[np.uint8]) -> str | None:
    """If many bright near-white pixels remain, prefer white over ground beige."""
    if region.size == 0 or region.ndim < 3:
        return None
    flat = region.reshape(-1, region.shape[-1])[:, :3].astype(np.float32)
    mean = flat.mean(axis=1)
    spread = flat.max(axis=1) - flat.min(axis=1)
    bright = (mean >= 175) & (spread <= 40)
    if int(bright.sum()) < max(6, int(0.12 * flat.shape[0])):
        return None
    return "white"


def dominant_color_for_entity(
    pixels: NDArray[np.uint8],
    box: BoundingBox,
    mask: SegmentationMask | None = None,
    *,
    label: str = "",
) -> str:
    """Entity-bound dominant color with background/vegetation/ground bleed rejection."""
    lab = (label or "").strip().lower()
    inset_x: float | None = None
    inset_y: float | None = None
    if lab in _SPORTS_SMALL:
        inset_x, inset_y = 0.22, 0.22
    elif lab in _APPLIANCE_LABELS:
        # Cabinets/walls often sit at bbox edges — sample the appliance body.
        inset_x, inset_y = 0.26, 0.20
    elif lab in _VEHICLE_LABELS:
        inset_x, inset_y = 0.16, 0.14
    elif lab in _ANIMAL_LABELS_COLOR:
        inset_x, inset_y = 0.14, 0.12

    region = _mask_pixels(pixels, box, mask, inset_x=inset_x, inset_y=inset_y)
    if lab in _VEHICLE_LABELS | _ANIMAL_LABELS_COLOR | _SPORTS_SMALL:
        region = exclude_vegetation_pixels(region)
    pre_earth = region
    if lab in _SPORTS_SMALL:
        region = exclude_earth_ground_pixels(region)
    if lab in _APPLIANCE_LABELS:
        # Drop warm wood/cabinet pixels that dominate loose fridge/oven boxes.
        region = exclude_earth_ground_pixels(region)

    estimate = estimate_color(region)
    if estimate.name == "unknown":
        if lab in _SPORTS_SMALL | _APPLIANCE_LABELS:
            white = _high_luminance_neutral_name(region) or _high_luminance_neutral_name(pre_earth)
            if white is not None:
                return white
        return "unknown"
    name = normalize_simple_color_name(estimate.name)

    # Low-confidence names are treated as unknown (honest refuse > wrong color).
    if estimate.confidence < 0.55 and lab not in _SPORTS_SMALL | _APPLIANCE_LABELS:
        return "unknown"

    if lab in _SPORTS_SMALL and name in _GROUND_TONES | {"unknown", "gray", "beige"}:
        white = _high_luminance_neutral_name(region) or _high_luminance_neutral_name(pre_earth)
        if white is not None:
            return white
        # Still ground-like after filtering — refuse rather than invent beige.
        if name in _GROUND_TONES | {"beige"}:
            return "unknown"

    if lab in _APPLIANCE_LABELS and name in _CABINET_WOOD_TONES | {"unknown", "gray"}:
        # Recoverable bright panel → white; never report cabinet wood as appliance color.
        white = _high_luminance_neutral_name(region) or _high_luminance_neutral_name(pre_earth)
        if white is not None:
            return white
        if name in _CABINET_WOOD_TONES:
            return "unknown"

    if lab in _VEHICLE_LABELS and name in {
        normalize_simple_color_name(t) for t in _FOLIAGE_TONES
    } | _FOLIAGE_TONES:
        return "unknown"

    if lab in _ANIMAL_LABELS_COLOR and name in _FOLIAGE_TONES | {
        normalize_simple_color_name(t) for t in _FOLIAGE_TONES
    }:
        return "unknown"

    return name if name != "unknown" else "unknown"


def _lab_chroma(lab: tuple[float, float, float]) -> float:
    return float((lab[1] * lab[1] + lab[2] * lab[2]) ** 0.5)


def _normalize_illumination(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Luminance lift only — never gray-world.

    Gray-world channel equalization was a primary root cause of chromatic→gray
    collapse (navy/red/brown cloth mapping to charcoal/light gray).
    """
    if rgb.size == 0:
        return rgb
    luminance = rgb.mean(axis=1, keepdims=True)
    lift = np.clip((40.0 - luminance) / 40.0, 0.0, 1.0) * 12.0
    return np.clip(rgb + lift, 0.0, 255.0).astype(np.float32)


def _rgb_to_lab(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    try:
        import cv2

        reshaped = rgb.reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        lab[:, 0] = lab[:, 0] * (100.0 / 255.0)
        lab[:, 1] = lab[:, 1] - 128.0
        lab[:, 2] = lab[:, 2] - 128.0
        return lab
    except Exception:  # noqa: BLE001
        r = rgb[:, 0] / 255.0
        g = rgb[:, 1] / 255.0
        b = rgb[:, 2] / 255.0
        l_val = 0.2126 * r + 0.7152 * g + 0.0722 * b
        a_val = (r - g) * 50.0
        b_val = (0.5 * (r + g) - b) * 50.0
        return np.stack([l_val * 100.0, a_val, b_val], axis=1).astype(np.float32)


def _rgb_to_hsv_mean(rgb: NDArray[np.float32]) -> tuple[float, float, float]:
    try:
        import cv2

        reshaped = rgb.reshape(-1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        mean = hsv.mean(axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])
    except Exception:  # noqa: BLE001
        mean = rgb.mean(axis=0)
        return 0.0, 0.0, float(mean.mean())


def _lab_distance(lab: tuple[float, float, float], ref: tuple[float, float, float]) -> float:
    """Chroma-weighted LAB distance (emphasize hue/chroma over lightness)."""
    dl = (lab[0] - ref[0]) * 0.7
    da = (lab[1] - ref[1]) * 1.2
    db = (lab[2] - ref[2]) * 1.2
    return float((dl * dl + da * da + db * db) ** 0.5)


def _nearest_chromatic_color(lab: tuple[float, float, float]) -> ColorEstimate | None:
    """Best non-gray named color when mid-chroma evidence argues against gray collapse."""
    best_name = ""
    best_dist = 1e9
    for name, ref in _NAMED_LAB:
        if name in _GRAY_NAMES | {"black", "white", "charcoal"}:
            continue
        dist = _lab_distance(lab, ref)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    # Allow muted navy/olive/burgundy/royal blue to recover before gray wins.
    if not best_name or best_dist > 36.0:
        return None
    confidence = float(max(0.0, min(1.0, 1.0 - best_dist / 54.0)))
    if confidence < 0.48:
        return None
    return ColorEstimate(best_name, confidence, lab)


def _nearest_named_color(lab: tuple[float, float, float]) -> ColorEstimate:
    best_name = "unknown"
    second_name = "unknown"
    best_dist = 1e9
    second_dist = 1e9
    chroma = _lab_chroma(lab)
    for name, ref in _NAMED_LAB:
        dist = _lab_distance(lab, ref)
        # Chromatic regions must not snap to the dense gray ladder.
        if chroma >= 5.0 and name in _GRAY_NAMES:
            dist *= 2.10
        if chroma >= 8.0 and name == "charcoal":
            dist *= 1.55
        if chroma >= 14.0 and name == "black":
            dist *= 1.35
        if dist < best_dist:
            second_dist = best_dist
            second_name = best_name
            best_dist = dist
            best_name = name
        elif dist < second_dist:
            second_dist = dist
            second_name = name
    # Require clear winner across families; same-family ties (olive/olive green) are fine.
    margin = second_dist - best_dist
    confidence = float(max(0.0, min(1.0, 1.0 - best_dist / 48.0)))
    if margin < 3.5 and not _same_color_family(best_name, second_name):
        confidence *= 0.72
    # Gray bias removal: any non-trivial chroma rejects gray unless chromatic recovery fails
    # AND confidence is extremely high on a near-achromatic sample.
    if best_name in _GRAY_NAMES:
        chromatic = _nearest_chromatic_color(lab)
        if chromatic is not None:
            return chromatic
        if chroma >= 4.5 or confidence < 0.88:
            return ColorEstimate("unknown", min(confidence, 0.48), lab)
    if best_name == "charcoal" and chroma >= 7.5:
        chromatic = _nearest_chromatic_color(lab)
        if chromatic is not None:
            return chromatic
        if confidence < 0.84:
            return ColorEstimate("unknown", min(confidence, 0.50), lab)
    # If a dense chromatic family still looks weak, prefer chromatic recovery over unknown.
    if confidence < _MIN_COLOR_CONFIDENCE and chroma >= 10.0:
        chromatic = _nearest_chromatic_color(lab)
        if chromatic is not None:
            return chromatic
    if confidence < _MIN_COLOR_CONFIDENCE:
        return ColorEstimate("unknown", confidence, lab)
    return ColorEstimate(best_name, confidence, lab)


def _robust_lab(normalized: NDArray[np.float32]) -> tuple[float, float, float]:
    """Median LAB after trimming extreme percentiles — resists outliers/background."""
    lab_vals = _rgb_to_lab(normalized)
    if lab_vals.shape[0] >= 24:
        low = np.percentile(lab_vals, 15, axis=0)
        high = np.percentile(lab_vals, 85, axis=0)
        mask = np.all((lab_vals >= low) & (lab_vals <= high), axis=1)
        trimmed = lab_vals[mask] if int(mask.sum()) >= 8 else lab_vals
    else:
        trimmed = lab_vals
    med = np.median(trimmed, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def estimate_color(region: NDArray[np.uint8]) -> ColorEstimate:
    """Estimate a natural color name from a pixel region with confidence."""
    if region.size == 0:
        return ColorEstimate("unknown", 0.0, (0.0, 0.0, 0.0))
    if region.ndim == 2:
        return ColorEstimate("unknown", 0.0, (0.0, 0.0, 0.0))
    flat = region.reshape(-1, region.shape[-1])[:, :3].astype(np.float32)
    if flat.shape[0] < 1:
        return ColorEstimate("unknown", 0.0, (0.0, 0.0, 0.0))
    region_std = float(flat.std())
    mixed_penalty = 0.22 if flat.shape[0] >= 16 and region_std > 65 else 0.0
    # Uniform patches are more trustworthy — slight confidence recovery.
    uniform_bonus = 0.06 if flat.shape[0] >= 16 and region_std < 28 else 0.0
    channel_spread = float(np.std(flat.mean(axis=0)))
    if flat.shape[0] >= 32 and region_std > 12 and channel_spread < 25:
        normalized = _normalize_illumination(flat)
    else:
        normalized = np.clip(flat, 0.0, 255.0).astype(np.float32)
    mean_lab = _robust_lab(normalized)
    estimate = _nearest_named_color(mean_lab)
    chroma = _lab_chroma(mean_lab)
    _, sat, _val = _rgb_to_hsv_mean(normalized)
    # Mixed fur/fabric: mean LAB can sit on the gray axis while a large chromatic
    # subset (brown horse, red jacket highlights) still carries the true hue.
    if (
        estimate.name in {"charcoal", "dark gray", "light gray", "black", "unknown"}
        and normalized.shape[0] >= 20
    ):
        lab_all = _rgb_to_lab(normalized)
        chroma_all = np.sqrt(lab_all[:, 1] ** 2 + lab_all[:, 2] ** 2)
        # Muted navy/olive jackets often sit near chroma 9–14 — still recoverable.
        chromatic_mask = chroma_all >= 9.0
        if int(chromatic_mask.sum()) >= max(6, int(0.16 * lab_all.shape[0])):
            med = np.median(lab_all[chromatic_mask], axis=0)
            chroma_subset = (float(med[0]), float(med[1]), float(med[2]))
            subset_est = _nearest_named_color(chroma_subset)
            if subset_est.name not in {"dark gray", "light gray", "unknown"}:
                estimate = subset_est
                mean_lab = chroma_subset
                chroma = _lab_chroma(mean_lab)
    # Prefer unknown / chromatic recovery over inventing gray.
    if chroma < 5.5 and estimate.name not in _STRICT_NEUTRALS | _GRAY_NAMES:
        if mean_lab[0] < 16:
            estimate = ColorEstimate("black", max(estimate.confidence, 0.76), estimate.lab)
        elif mean_lab[0] < 26:
            estimate = ColorEstimate("charcoal", max(estimate.confidence, 0.74), estimate.lab)
        elif mean_lab[0] > 88:
            estimate = ColorEstimate("white", max(estimate.confidence, 0.72), estimate.lab)
        elif mean_lab[0] > 80 and mean_lab[2] > 4:
            estimate = ColorEstimate("cream", max(estimate.confidence, 0.68), estimate.lab)
        else:
            # Uncertain mid-achromatic → unknown (never light/dark gray by default).
            estimate = ColorEstimate("unknown", 0.40, estimate.lab)
    elif 5.5 <= chroma < 16.0 and estimate.name in _GRAY_NAMES | {"charcoal"}:
        chromatic = _nearest_chromatic_color(mean_lab)
        if chromatic is not None and chromatic.confidence >= 0.48:
            estimate = chromatic
        elif mean_lab[2] > 6.0 and mean_lab[0] > 62:
            if mean_lab[0] > 85:
                warm = "cream"
            elif mean_lab[2] > 22 and mean_lab[0] < 72:
                warm = "tan"
            elif mean_lab[1] < 0 and mean_lab[2] > 18:
                warm = "khaki"
            else:
                warm = "beige"
            estimate = ColorEstimate(warm, max(0.58, estimate.confidence * 0.9), estimate.lab)
        elif mean_lab[0] < 20 and chroma < 7.0:
            estimate = ColorEstimate("black", max(0.64, estimate.confidence * 0.9), estimate.lab)
        elif mean_lab[0] < 36 and abs(mean_lab[1]) + abs(mean_lab[2]) >= 10:
            if mean_lab[2] < -10:
                estimate = ColorEstimate("navy blue", max(0.60, estimate.confidence * 0.9), estimate.lab)
            elif mean_lab[2] < -4 and mean_lab[1] > 8:
                estimate = ColorEstimate("royal blue", max(0.58, estimate.confidence * 0.9), estimate.lab)
            elif mean_lab[1] < -6 and mean_lab[2] > 4:
                estimate = ColorEstimate(
                    "olive" if mean_lab[0] > 36 else "forest green",
                    max(0.58, estimate.confidence * 0.9),
                    estimate.lab,
                )
            elif mean_lab[1] > 10 and mean_lab[2] > 4:
                estimate = ColorEstimate("brown", max(0.58, estimate.confidence * 0.9), estimate.lab)
            else:
                estimate = ColorEstimate("unknown", 0.46, estimate.lab)
        else:
            estimate = ColorEstimate("unknown", 0.44, estimate.lab)
    elif chroma >= 9.0 and estimate.name in _GRAY_NAMES | {"charcoal"}:
        chromatic = _nearest_chromatic_color(mean_lab)
        estimate = chromatic if chromatic is not None else ColorEstimate("unknown", 0.45, mean_lab)
    elif sat >= 24 and chroma >= 7.0 and estimate.name in _GRAY_NAMES | {"charcoal", "black"}:
        chromatic = _nearest_chromatic_color(mean_lab)
        if chromatic is not None:
            estimate = chromatic
        elif estimate.name in _GRAY_NAMES:
            estimate = ColorEstimate("unknown", 0.45, estimate.lab)
    # Final hard gate: never emit gray without near-zero chroma + high confidence.
    if estimate.name in _GRAY_NAMES and (chroma >= 4.0 or estimate.confidence < 0.86):
        estimate = ColorEstimate("unknown", min(estimate.confidence, 0.48), estimate.lab)
    confidence = max(0.0, min(1.0, estimate.confidence - mixed_penalty + uniform_bonus))
    if flat.shape[0] == 1:
        confidence = max(confidence, 0.65 if estimate.name != "unknown" else 0.0)
    if chroma >= 14.0 and estimate.name not in _STRICT_NEUTRALS | _GRAY_NAMES:
        confidence = max(confidence, 0.62)
    min_conf = _MIN_COLOR_CONFIDENCE + (0.05 if mixed_penalty > 0 and chroma < 12.0 else 0.0)
    if estimate.name in _GRAY_NAMES:
        min_conf = max(min_conf, 0.84)
    if confidence < min_conf or estimate.name == "unknown":
        return ColorEstimate("unknown", confidence, estimate.lab)
    return ColorEstimate(estimate.name, confidence, estimate.lab)


def rgb_to_color_name(r: float, g: float, b: float) -> str:
    """Map a single RGB triple to a natural color name (legacy helper)."""
    region = np.full((8, 8, 3), (r, g, b), dtype=np.uint8)
    return estimate_color(region).name


def dominant_color_name(
    pixels: NDArray[np.uint8],
    box: BoundingBox,
    mask: SegmentationMask | None = None,
    *,
    label: str = "",
) -> str:
    """Dominant natural color from mask/bbox; unknown when uncertain.

    When ``label`` is provided, use entity-aware bleed rejection
    (vegetation/ground inset + simple vocabulary normalization).
    """
    if label:
        return dominant_color_for_entity(pixels, box, mask, label=label)
    return normalize_simple_color_name(estimate_color(_mask_pixels(pixels, box, mask)).name)


def secondary_color_name(
    pixels: NDArray[np.uint8],
    box: BoundingBox,
    mask: SegmentationMask | None = None,
    *,
    label: str = "",
) -> str:
    """Secondary color from lower sampled region."""
    lab = (label or "").strip().lower()
    inset_x = inset_y = None
    if lab in _SPORTS_SMALL:
        inset_x, inset_y = 0.22, 0.22
    elif lab in _APPLIANCE_LABELS:
        inset_x, inset_y = 0.26, 0.20
    elif lab in _VEHICLE_LABELS:
        inset_x, inset_y = 0.16, 0.14
    region = _mask_pixels(pixels, box, mask, inset_x=inset_x, inset_y=inset_y)
    if lab in _VEHICLE_LABELS | _ANIMAL_LABELS_COLOR | _SPORTS_SMALL:
        region = exclude_vegetation_pixels(region)
    if lab in _SPORTS_SMALL | _APPLIANCE_LABELS:
        region = exclude_earth_ground_pixels(region)
    if region.size == 0 or region.shape[0] < 4:
        return dominant_color_name(pixels, box, mask, label=label)
    lower = region[region.shape[0] // 2 :, :]
    name = normalize_simple_color_name(estimate_color(lower).name)
    primary = dominant_color_name(pixels, box, mask, label=label)
    if name == primary or name == "unknown":
        return "unknown"
    if lab in _VEHICLE_LABELS | _ANIMAL_LABELS_COLOR and name in _FOLIAGE_TONES | {
        normalize_simple_color_name(t) for t in _FOLIAGE_TONES
    }:
        return "unknown"
    if lab in _SPORTS_SMALL | _APPLIANCE_LABELS and name in _GROUND_TONES | _CABINET_WOOD_TONES:
        return "unknown"
    return name


def region_color_name(
    pixels: NDArray[np.uint8],
    box: BoundingBox,
    *,
    vertical_band: str,
    mask: SegmentationMask | None = None,
    exclude_skin: bool = False,
) -> str:
    """Sample color from top/middle/bottom garment band with mask awareness.

    Uses center-biased inset bands of the person box — never the whole body crop.
    """
    if vertical_band == "top":
        # Hair / head — avoid shoulders and background sides.
        band_box = BoundingBox(
            box.x_min + box.width * 0.22,
            box.y_min + box.height * 0.02,
            box.x_max - box.width * 0.22,
            box.y_min + max(1.0, box.height * 0.16),
        )
    elif vertical_band == "bottom":
        # Pants / lower garment — above footwear, inset from sides/ground.
        band_box = BoundingBox(
            box.x_min + box.width * 0.18,
            box.y_min + box.height * 0.58,
            box.x_max - box.width * 0.18,
            box.y_min + box.height * 0.82,
        )
    else:
        # Shirt / upper torso — center-biased, below head, above waist.
        start = box.y_min + box.height * 0.28
        band_box = BoundingBox(
            box.x_min + box.width * 0.20,
            start,
            box.x_max - box.width * 0.20,
            start + max(1.0, box.height * 0.26),
        )
    region = _mask_pixels(pixels, band_box, mask, inset_x=0.10, inset_y=0.08)
    if exclude_skin:
        region = exclude_skin_pixels(region)
        # Grass / foliage must not become shirt or pants color.
        region = exclude_vegetation_pixels(region)
        if vertical_band == "bottom":
            # Dirt/road under legs must not become pants color.
            region = exclude_earth_ground_pixels(region)
    estimate = estimate_color(region)
    if estimate.confidence < 0.55 or estimate.name == "unknown":
        return "unknown"
    name = normalize_simple_color_name(estimate.name)
    if exclude_skin and name in _FOLIAGE_TONES | {
        normalize_simple_color_name(t) for t in _FOLIAGE_TONES
    }:
        return "unknown"
    if exclude_skin and vertical_band == "bottom" and name in {"beige", "tan", "khaki", "cream"}:
        # Weak warm neutrals on lower body are usually ground bleed.
        if estimate.confidence < 0.72:
            return "unknown"
    return name


def scene_color_palette(pixels: NDArray[np.uint8], max_colors: int = 8) -> tuple[str, ...]:
    """Extract high-confidence dominant scene colors."""
    if pixels.size == 0:
        return ()
    sample = pixels[:: max(1, pixels.shape[0] // 64), :: max(1, pixels.shape[1] // 64)]
    flat = sample.reshape(-1, sample.shape[2])[:, :3]
    if flat.size == 0:
        return ()
    buckets: dict[str, int] = {}
    for rgb in flat[:: max(1, len(flat) // 400)]:
        estimate = estimate_color(np.array([[rgb]], dtype=np.uint8))
        if estimate.name == "unknown":
            continue
        buckets[estimate.name] = buckets.get(estimate.name, 0) + 1
    ranked = sorted(buckets.items(), key=lambda item: -item[1])
    return tuple(name for name, _ in ranked[:max_colors])
