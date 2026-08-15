"""Caption detail / grammar regression helpers for street-style scenes."""

from __future__ import annotations

from language.refinement.caption_sanity import caption_sanity_score, sanitize_caption


def test_detailed_caption_keeps_colors_and_drops_robotic_filler() -> None:
    draft = (
        "A person in a dark gray jacket is crossing street on a city street. "
        "A maroon car is also visible in the scene. "
        "Around them lies a city street, with trees lining the edge of the view."
    )
    cleaned = sanitize_caption(draft)
    assert "dark gray" in cleaned.lower()
    assert "maroon" in cleaned.lower()
    assert "crossing a" in cleaned.lower()
    assert "around them lies" not in cleaned.lower()
    assert "trees" in cleaned.lower()
    # Should remain a coherent multi-sentence paragraph, not a single stub.
    assert len(cleaned.split()) >= 20
    assert "crossing street on" not in cleaned.lower()
