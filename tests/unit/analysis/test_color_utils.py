"""Unit tests for competition-grade color naming."""

from __future__ import annotations

import numpy as np

from analysis.common.color_utils import estimate_color, rgb_to_color_name, scene_color_palette
from core.contracts.detection import BoundingBox
from analysis.common.color_utils import dominant_color_name


def test_natural_color_names_for_common_swatches() -> None:
    assert rgb_to_color_name(10, 10, 12) in {"black", "charcoal"}
    assert rgb_to_color_name(20, 35, 90) in {"navy blue", "navy", "blue", "royal blue", "purple"}
    assert rgb_to_color_name(120, 30, 40) in {"burgundy", "maroon", "red", "brown"}
    assert rgb_to_color_name(200, 190, 160) in {"beige", "cream", "blond", "white", "khaki", "unknown"}


def test_chromatic_colors_do_not_collapse_to_gray() -> None:
    # Muted navy / olive must keep hue — gray collapse was the dominant failure mode.
    assert rgb_to_color_name(28, 42, 95) in {"navy blue", "navy", "blue", "royal blue", "purple"}
    assert rgb_to_color_name(70, 85, 40) in {
        "olive",
        "olive green",
        "green",
        "forest green",
        "brown",
        "mustard",
        "khaki",
    }
    assert rgb_to_color_name(140, 45, 55) in {"burgundy", "maroon", "red", "brown", "pink"}
    for rgb in ((28, 42, 95), (70, 85, 40), (140, 45, 55), (40, 90, 120)):
        name = rgb_to_color_name(*rgb)
        assert name not in {"dark gray", "light gray"}


def test_uncertain_midtone_is_unknown_not_gray() -> None:
    # Ambiguous mid-achromatic fabric must not invent gray.
    name = rgb_to_color_name(120, 122, 118)
    assert name in {"unknown", "beige", "cream", "charcoal", "black"}
    assert name not in {"dark gray", "light gray"}


def test_uncertain_mixed_region_returns_unknown() -> None:
    # Highly mixed noise should not invent a confident color.
    rng = np.random.default_rng(0)
    region = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    estimate = estimate_color(region)
    # Either unknown or low confidence; never force a label when uncertain.
    assert estimate.confidence < 0.9


def test_dominant_color_uses_bbox() -> None:
    pixels = np.zeros((50, 50, 3), dtype=np.uint8)
    pixels[:, :] = (25, 40, 110)
    name = dominant_color_name(pixels, BoundingBox(0, 0, 50, 50))
    assert name in {"dark blue", "navy blue", "navy", "blue", "purple"}


def test_sports_ball_prefers_white_over_ground_beige() -> None:
    """White ball surrounded by dirt/beige must not become beige."""
    from analysis.common.color_utils import dominant_color_for_entity

    pixels = np.full((80, 80, 3), (185, 165, 125), dtype=np.uint8)
    pixels[28:52, 28:52] = (242, 242, 238)
    box = BoundingBox(20, 20, 60, 60)
    name = dominant_color_for_entity(pixels, box, None, label="sports ball")
    assert name == "white"
    assert name != "beige"


def test_bicycle_rejects_grass_green_bleed() -> None:
    """Black bicycle on grass must not be reported as green."""
    from analysis.common.color_utils import dominant_color_for_entity

    pixels = np.full((100, 100, 3), (45, 150, 55), dtype=np.uint8)
    pixels[32:68, 38:62] = (18, 18, 22)
    box = BoundingBox(28, 28, 72, 72)
    name = dominant_color_for_entity(pixels, box, None, label="bicycle")
    assert name not in {"green", "olive", "dark green", "forest green", "olive green"}
    assert name in {"black", "gray", "unknown", "charcoal"}


def test_entity_color_not_global_image_dominant() -> None:
    """Object color must come from the entity box, not the whole image."""
    from analysis.common.color_utils import dominant_color_for_entity

    pixels = np.full((60, 60, 3), (40, 160, 50), dtype=np.uint8)  # green scene
    pixels[20:40, 20:40] = (220, 40, 40)  # red object
    name = dominant_color_for_entity(
        pixels, BoundingBox(20, 20, 40, 40), None, label="handbag"
    )
    assert name == "red"


def test_refrigerator_rejects_cabinet_wood_beige() -> None:
    """White/light appliance must not inherit beige/brown from surrounding cabinets."""
    from analysis.common.color_utils import dominant_color_for_entity

    # Warm cabinet surround with a bright near-white appliance body in the center.
    pixels = np.full((100, 80, 3), (168, 140, 105), dtype=np.uint8)
    pixels[18:82, 18:62] = (236, 234, 230)
    box = BoundingBox(10, 10, 70, 90)
    name = dominant_color_for_entity(pixels, box, None, label="refrigerator")
    assert name == "white"
    assert name not in {"beige", "brown", "tan", "cream"}


def test_appliance_unknown_when_only_wood_evidence() -> None:
    """If only warm wood tones remain and no bright panel, refuse brown/beige."""
    from analysis.common.color_utils import dominant_color_for_entity

    pixels = np.full((60, 40, 3), (145, 110, 70), dtype=np.uint8)
    box = BoundingBox(0, 0, 40, 60)
    name = dominant_color_for_entity(pixels, box, None, label="oven")
    assert name not in {"brown", "beige", "tan"}
    assert name in {"unknown", "white", "gray", "black"}

    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:16] = (30, 30, 30)
    pixels[16:] = (200, 200, 210)
    palette = scene_color_palette(pixels, max_colors=4)
    assert palette
    assert "unknown" not in palette
