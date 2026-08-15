"""Caption naturalness polish — style only, no evidence/QA changes."""

from __future__ import annotations

from language.refinement.caption_sanity import humanize_caption_style, sanitize_caption
from language.semantic.natural_caption_service import NaturalCaptionService

_ROBOTIC = (
    "situated",
    "positioned within",
    "occupies",
    "participates in",
    "indicating",
    "suggesting",
    "transportation corridor",
    "recreational setting",
    "natural environment",
    "environment setting",
    "located in proximity",
    "spatially related",
    "are an activity",
    "is an activity",
    "actively",
)


def _assert_not_robotic(text: str) -> None:
    lower = text.lower()
    for phrase in _ROBOTIC:
        assert phrase not in lower, f"robotic phrase left behind: {phrase!r} in {text!r}"


def test_horse_person_fire_scene_rejects_activity_template() -> None:
    raw = (
        "Two people are engaged in an activity within a grassy field. "
        "One person, dressed in khaki attire - a khaki shirt and red pants "
        "paired with beige shoes - is actively leading a brown horse. "
        "Smoke and fire are visible nearby."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "are an activity" not in lower
    assert "engaged in an activity" not in lower
    assert "within a" not in lower
    assert "attire" not in lower
    assert "actively" not in lower
    assert "two people" in lower
    assert "grassy field" in lower
    assert "khaki shirt" in lower
    assert "red pants" in lower
    assert "brown horse" in lower
    assert "leading" in lower or "holding" in lower
    assert "fire" in lower or "smoke" in lower
    # Coherent scene prose, not inventory spam.
    assert lower.count(",") < 8


def test_motorcycle_rejects_transportation_corridor_style() -> None:
    raw = (
        "The motorcycle is situated in a transportation corridor setting, "
        "indicating it's part of an outdoor traffic flow."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "motorcycle" in lower
    assert "road" in lower
    assert "is on a road" in lower or "on a road" in lower


def test_motorcycle_riding_preserves_verified_action() -> None:
    raw = "A person is riding a motorcycle on a road. The scene indicates part of an outdoor traffic flow."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "person" in lower
    assert "riding" in lower
    assert "motorcycle" in lower
    assert "road" in lower


def test_kitchen_rejects_abstract_environment_phrasing() -> None:
    raw = "The environment indicates a domestic food preparation setting."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "environment indicates" not in lower
    assert "domestic food preparation setting" not in lower
    assert "kitchen" in lower
    assert len(cleaned.split()) >= 2


def test_kitchen_accepts_simple_human_description() -> None:
    raw = "A person is in a kitchen near a dining table."
    cleaned = sanitize_caption(raw)
    assert cleaned.lower().startswith("a person is in a kitchen")
    assert "dining table" in cleaned.lower()


def test_kitchen_preserves_colors_and_relations() -> None:
    raw = (
        "A person wearing a blue shirt stands near a white refrigerator in a kitchen. "
        "A cup is on the table."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "blue shirt" in lower
    assert "refrigerator" in lower
    assert "cup" in lower
    assert "table" in lower
    assert "kitchen" in lower


def test_animal_rejects_occupies_natural_environment() -> None:
    raw = "The animal occupies a natural environment."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "occupies" not in lower
    assert "natural environment" not in lower


def test_animal_accepts_simple_bear_description() -> None:
    raw = "A bear stands in a grassy area."
    cleaned = sanitize_caption(raw)
    assert "bear" in cleaned.lower()
    assert "grassy" in cleaned.lower()
    assert "occupies" not in cleaned.lower()


def test_animal_preserves_spatial_relation() -> None:
    raw = "A brown dog is standing on grass near a wooden fence."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "brown dog" in lower
    assert "grass" in lower
    assert "fence" in lower
    _assert_not_robotic(cleaned)


def test_vehicle_rejects_urban_transportation_report() -> None:
    raw = "The vehicle participates in urban transportation."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    _assert_not_robotic(cleaned)
    assert "participates" not in lower
    assert "urban transportation" not in lower
    assert "road" in lower


def test_vehicle_accepts_simple_bus_description() -> None:
    raw = "A bus is parked on a road."
    cleaned = sanitize_caption(raw)
    assert cleaned.lower().startswith("a bus is parked on a road")


def test_vehicle_preserves_color_and_action() -> None:
    raw = "A red car is parked beside a blue bus on a city street."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "red car" in lower
    assert "blue bus" in lower
    assert "parked" in lower
    _assert_not_robotic(cleaned)


def test_humanize_preserves_verified_facts() -> None:
    raw = "A person is riding a red motorcycle on a road."
    cleaned = humanize_caption_style(raw)
    lower = cleaned.lower()
    assert "person" in lower
    assert "riding" in lower
    assert "motorcycle" in lower
    assert "road" in lower
    assert "red" in lower


def test_relation_robotic_phrases_rewritten() -> None:
    raw = (
        "The cup is located in proximity to the table. "
        "The chair is positioned adjacent to the desk."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "located in proximity" not in lower
    assert "positioned adjacent" not in lower
    assert "near" in lower or "beside" in lower
    assert "cup" in lower
    assert "table" in lower


def test_sits_within_the_scene_rewritten() -> None:
    raw = "A vase sits within the scene. A clock sits within the scene."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "sits within" not in lower
    assert "vase" in lower
    assert "nearby" in lower or "clock" in lower


def test_sanitize_place_maps_transportation_corridor() -> None:
    assert NaturalCaptionService._sanitize_place("transportation corridor") == "road"
    assert NaturalCaptionService._sanitize_place("natural environment") == "outdoors"
    assert NaturalCaptionService._sanitize_place("kitchen") == "kitchen"
