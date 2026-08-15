"""Deterministic caption sanity: grammar, filler, dedupe, baseball regression."""

from __future__ import annotations

from language.refinement.caption_sanity import (
    caption_sanity_score,
    choose_better_caption,
    dedupe_object_mention_sentences,
    fix_double_articles,
    has_awkward_filler,
    sanitize_caption,
    strip_spatial_filler,
)
from language.semantic.natural_caption_service import NaturalCaptionService


def test_fix_double_articles() -> None:
    assert "an a" not in fix_double_articles("stands near an a baseball glove").lower()
    assert "a a" not in fix_double_articles("a a person stands").lower()


def test_strip_spatial_filler() -> None:
    text = (
        "A person stands near a baseball glove. "
        "In the foreground, a baseball glove sits close enough to matter to the action."
    )
    cleaned = strip_spatial_filler(text)
    assert "close enough to matter" not in cleaned.lower()
    assert "matter to the action" not in cleaned.lower()


def test_dedupe_repeated_object_sentences() -> None:
    text = (
        "A person stands near a baseball glove. "
        "In the foreground, a baseball glove sits close enough to matter to the action."
    )
    cleaned = sanitize_caption(text)
    assert cleaned.lower().count("baseball glove") <= 1
    assert "an a" not in cleaned.lower()
    assert "close enough to matter" not in cleaned.lower()


def test_awkward_someone_nearby_farther_back() -> None:
    text = "Someone nearby farther back wears sneakers."
    assert has_awkward_filler(text)
    cleaned = sanitize_caption(text)
    assert "someone nearby farther back" not in cleaned.lower()


def test_article_never_double_wraps() -> None:
    service = NaturalCaptionService.__new__(NaturalCaptionService)
    assert service._article("a baseball glove") == "a"
    phrase = f"{service._article('a baseball glove')} {service._bare_phrase('a baseball glove')}"
    assert phrase == "a baseball glove"


def test_baseball_regression_semantic_quality() -> None:
    bad = (
        "A person stands near an a baseball glove. "
        "Someone nearby farther back wears sneakers. "
        "In the foreground, a baseball glove sits close enough to matter to the action."
    )
    better = (
        "Two people stand on a baseball field. "
        "A baseball glove is visible near one person, and another person farther back wears sneakers."
    )
    chosen = choose_better_caption(bad, better)
    assert "an a" not in chosen.lower()
    assert "close enough to matter" not in chosen.lower()
    assert "someone nearby farther back" not in chosen.lower()
    assert "baseball glove" in chosen.lower()
    assert chosen.lower().count("baseball glove") <= 1
    assert caption_sanity_score(chosen) > caption_sanity_score(bad)
    # One paragraph (no forced multi-block structure requirement beyond joined sentences).
    assert "\n\n" not in chosen


def test_sanitize_removes_robotic_inventory_phrasing() -> None:
    text = (
        "A beige bowl and a brown bowl and a sink are also visible in the scene. "
        "Up close, the action is kitchen preparation. Is also visible."
    )
    cleaned = sanitize_caption(text)
    assert "also visible" not in cleaned.lower()
    assert "up close, the action" not in cleaned.lower()
    assert "kitchen preparation" in cleaned.lower()
    assert "bowl" in cleaned.lower()


def test_sanitize_workstation_bad_caption_regression() -> None:
    text = (
        "A person is working at a computer in an office workspace. "
        "The main work underway is using a keyboard. "
        "A charcoal tv and a navy chair share the surrounding space. "
        "They are using a keyboard. "
        "The setting remains clearly indoors."
    )
    cleaned = sanitize_caption(text)
    assert cleaned.lower().count("keyboard") <= 1
    assert "main work underway" not in cleaned.lower()
    assert "setting remains" not in cleaned.lower()
    assert "share the surrounding" not in cleaned.lower()
    assert "they are using" not in cleaned.lower()


def test_choose_better_rejects_grazing_fragment() -> None:
    bad = (
        "A couch defines this living room interior. Close beside a pink cat, "
        "while a cat grazes quietly farther back. Is also visible."
    )
    better = (
        "Two cats rest on a living-room couch. One pink cat lies close beside "
        "the cushions while another cat sits farther back indoors."
    )
    chosen = choose_better_caption(bad, better)
    assert "grazes" not in chosen.lower()
    assert "is also visible" not in chosen.lower()
    assert "cat" in chosen.lower()


def test_dedupe_keeps_distinct_facts() -> None:
    text = "A person prepares food in a kitchen. A sink and a bowl are nearby."
    cleaned = dedupe_object_mention_sentences(text)
    assert "person" in cleaned.lower()
    assert "kitchen" in cleaned.lower() or "sink" in cleaned.lower()


def test_activity_not_shared_across_two_people() -> None:
    bad = (
        "The overall impression suggests a casual moment on the farm with "
        "two people engaged in leading a horse."
    )
    cleaned = sanitize_caption(bad)
    lower = cleaned.lower()
    assert "two people engaged in leading" not in lower
    assert "overall impression" not in lower
    assert "casual moment" not in lower


def test_proximity_does_not_become_participation() -> None:
    text = (
        "One person is leading a horse while another person is observing the activity."
    )
    cleaned = sanitize_caption(text)
    lower = cleaned.lower()
    assert "observing the activity" not in lower
    assert "leading" in lower


def test_strip_unsupported_subjective_language() -> None:
    text = (
        "A pair of individuals are present in an outdoor farm pasture setting. "
        "The overall impression suggests a casual moment."
    )
    cleaned = sanitize_caption(text)
    lower = cleaned.lower()
    assert "pair of individuals" not in lower
    assert "outdoor farm pasture setting" not in lower
    assert "overall impression" not in lower
    assert "casual moment" not in lower
    assert "two people" in lower or "people" in lower


def test_unreliable_animal_color_fallback() -> None:
    from language.refinement.caption_sanity import normalize_animal_coat_color

    assert normalize_animal_coat_color("olive") == "brown"
    assert normalize_animal_coat_color("burgundy") == "brown"
    cleaned = sanitize_caption("A person is leading an olive horse across the field.")
    assert "olive" not in cleaned.lower()
    assert "brown horse" in cleaned.lower()
    colored = sanitize_caption("One person is leading a khaki-colored horse.")
    assert "khaki" not in colored.lower()
    assert "tan horse" in colored.lower()


def test_caption_duplicate_leading_removed() -> None:
    text = (
        "One person is leading a horse. "
        "The person is engaged in leading the horse."
    )
    cleaned = sanitize_caption(text)
    assert cleaned.lower().count("leading") == 1


def test_robotic_scene_zone_phrases_humanized() -> None:
    cleaned = sanitize_caption(
        "The white sports ball rests at the bottom center of the scene. "
        "Two people are with a brown horse in outdoors. "
        "She leads a brown horse across an outdoors."
    )
    lower = cleaned.lower()
    assert "bottom center" not in lower
    assert "in outdoors" not in lower
    assert "an outdoors" not in lower
    assert "outdoors" in lower or "nearby" in lower


def test_natural_caption_rejects_unsafe_animal_coat() -> None:
    service = NaturalCaptionService.__new__(NaturalCaptionService)
    assert service._color_claim_allowed("olive", 0.9, person=False, animal=True)
    assert service._normalize_animal_coat_color("olive") == "brown"
    assert not service._color_claim_allowed("purple", 0.4, person=False, animal=True)
