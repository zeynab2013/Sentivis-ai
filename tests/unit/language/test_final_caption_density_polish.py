"""Final polish: richer factual captions without padding or hallucination."""

from __future__ import annotations

from language.refinement.caption_coverage import (
    expand_verified_information_density,
    ensure_salient_verified_coverage,
    strip_redundant_caption_sentences,
)
from language.semantic.narrative_generator import (
    executive_summary_from_paragraph,
    short_caption_from_paragraph,
)

RICH_PACKED = (
    "On a grassy field, a person in a black sweatshirt holds a rope while leading a large "
    "brown horse, while another person and another horse stand farther back in the field. "
    "In the foreground, a fire burns."
)

SPARSE = "A person stands beside a table in an indoor room."


def test_rich_scene_expands_to_more_detailed_caption() -> None:
    expanded = expand_verified_information_density(RICH_PACKED)
    sentences = [s for s in expanded.split(".") if s.strip()]
    assert len(sentences) >= 3
    lower = expanded.lower()
    assert "black sweatshirt" in lower
    assert "leading" in lower
    assert "rope" in lower
    assert "farther back" in lower or "second horse" in lower
    assert "fire" in lower
    assert "smoke" not in lower
    assert "khaki" not in lower
    assert "closer to the camera" not in lower
    # No redundant headcount closer.
    assert "two people and two horses are visible" not in lower


def test_sparse_scene_not_artificially_padded() -> None:
    expanded = expand_verified_information_density(SPARSE)
    assert expanded.count(".") <= 2
    assert "two people" not in expanded.lower()
    assert "fire" not in expanded.lower()
    assert "spread across" not in expanded.lower()
    assert len(expanded.split()) <= len(SPARSE.split()) + 4


def test_expanded_caption_stays_factual_with_coverage() -> None:
    expanded = expand_verified_information_density(RICH_PACKED)
    covered = ensure_salient_verified_coverage(
        expanded,
        environment_evidence=("Hazard detected: fire (confidence: 88%)",),
    )
    assert "fire" in covered.lower()
    assert "smoke" not in covered.lower()


def test_short_and_executive_still_valid_after_enrichment() -> None:
    expanded = expand_verified_information_density(RICH_PACKED)
    short = short_caption_from_paragraph(expanded)
    executive = executive_summary_from_paragraph(expanded)
    assert short.endswith((".", "!", "?"))
    assert not short.endswith("...")
    assert executive.strip() != expanded.strip()
    assert short.strip() != expanded.strip()
    assert "while another..." not in short.lower()
    assert short.count(".") <= 2
    # Short should remain complete and grounded.
    assert "horse" in short.lower()
    assert "fire" in short.lower() or "leading" in short.lower()


def test_strip_redundant_count_sentence() -> None:
    text = (
        "On a grassy field, a person wearing a black sweatshirt holds a rope while leading "
        "a large brown horse. Farther back in the field, another person stands near a second "
        "horse. A fire is burning in the foreground. Two people and two horses are visible "
        "across the open field."
    )
    cleaned = strip_redundant_caption_sentences(text)
    assert "two people and two horses are visible" not in cleaned.lower()
    assert "black sweatshirt" in cleaned.lower()
    assert "fire" in cleaned.lower()


def test_distribution_sentence_not_count_padding() -> None:
    expanded = expand_verified_information_density(RICH_PACKED)
    lower = expanded.lower()
    assert "two people and two horses are visible" not in lower
    # Density expansion restructures packed clauses — no category summary padding.
    assert "spread across the open outdoor setting" not in lower
    assert len([s for s in expanded.split(".") if s.strip()]) >= 3
    assert "farther back" in lower
    assert "fire" in lower