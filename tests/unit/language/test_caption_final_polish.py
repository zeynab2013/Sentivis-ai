"""Final polish: malformed fragments, ride/activity dedupe, support-object fold."""

from __future__ import annotations

from language.refinement.caption_coverage import _activity_already_expressed
from language.refinement.caption_sanity import sanitize_caption


def test_motorcycle_drops_duplicate_riding_and_malformed_fragment() -> None:
    raw = (
        "A person is riding a dirt bike. "
        "A person, gloves and riding a bike on the water. "
        "Large rocks and grass on the ground next to the water. "
        "A brown motorcycle is visible behind them. "
        "A person is riding. A person is riding."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "riding a dirt bike" in lower or "motorcycle" in lower
    assert "on the water" not in lower
    assert "gloves and riding" not in lower
    assert lower.count("a person is riding.") == 0
    assert "riding. a person is riding" not in lower


def test_baseball_drops_malformed_they_are_fragment() -> None:
    raw = (
        "A young person is swinging a baseball bat at a white ball. "
        "They are, a blue shirt and black pants."
    )
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "swinging" in lower
    assert "baseball bat" in lower or "ball" in lower
    assert "they are," not in lower


def test_soccer_folds_ball_support_into_activity() -> None:
    raw = "Two people are playing football. A white sports ball rests in the scene."
    cleaned = sanitize_caption(raw)
    lower = cleaned.lower()
    assert "two people are playing football" in lower
    assert "while" in lower
    assert "ball" in lower
    assert "rests in the scene" not in lower
    # Actor ownership must remain two, never inflate.
    assert "four people" not in lower
    assert not lower.startswith("four")


def test_rope_hold_still_deduped() -> None:
    raw = (
        "A person wearing light clothing holds a rope while leading a brown horse outdoors. "
        "A person is holding a rope. A fire is burning nearby."
    )
    cleaned = sanitize_caption(raw).lower()
    assert cleaned.count("holding a rope") + cleaned.count("holds a rope") == 1
    assert "leading" in cleaned
    assert "fire" in cleaned


def test_riding_motorcycle_covered_by_dirt_bike_phrasing() -> None:
    assert _activity_already_expressed(
        "a person is riding a dirt bike outdoors.",
        "riding a motorcycle",
    )
    assert not _activity_already_expressed(
        "a person is riding a dirt bike outdoors.",
        "riding a bicycle",
    )


def test_they_leads_grammar_fix() -> None:
    raw = "A person is on the grass as they leads one of the horses."
    cleaned = sanitize_caption(raw).lower()
    assert "they lead" in cleaned
    assert "they leads" not in cleaned


def test_they_guides_grammar_fix() -> None:
    raw = "They are leading a horse as they guides it across the grass."
    cleaned = sanitize_caption(raw).lower()
    assert "they guide" in cleaned
    assert "they guides" not in cleaned


def test_positioned_near_is_humanized() -> None:
    raw = "The bicycle is positioned near the person."
    cleaned = sanitize_caption(raw).lower()
    assert "positioned near" not in cleaned
    assert "is near" in cleaned
