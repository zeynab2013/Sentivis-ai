"""Naturalness guards: gender neutralization must not produce robotic prose."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
from core.contracts.analysis import (
    ActivityHints,
    Attribute,
    AttributeSet,
    EnvironmentInfo,
    Relation,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from language.refinement.caption_coverage import ensure_salient_verified_coverage
from language.refinement.caption_sanity import humanize_caption_style, sanitize_caption
from language.validation.caption_factuality import filter_unsupported_claims_verified


def _verified_two_people():
    nodes = (
        SceneNode(0, "person_1", "person", 0.3, "middle-left"),
        SceneNode(1, "person_2", "person", 0.25, "middle-right"),
        SceneNode(2, "refrigerator_1", "refrigerator", 0.2, "middle-center"),
    )
    env = EnvironmentInfo(
        scene_type="kitchen",
        setting="kitchen",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="indoor",
        social_context="",
        crowd_level="few",
        scene_complexity="medium",
        evidence=("Kitchen appliances visible.",),
    )
    ctx = SceneContext(
        graph=SceneGraph(nodes=nodes, relations=(Relation(0, 2, "near", 0.7),)),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(1, "confidence", "88%"),
                Attribute(2, "dominant_color", "white"),
                Attribute(2, "confidence", "91%"),
            )
        ),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=env,
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes),
        spatial_summary="",
    )
    return build_verified_scene_evidence(ctx)


def test_gender_neutralize_becomes_two_people_not_person_and_person() -> None:
    verified = _verified_two_people()
    text = "A man and woman are present in an indoor kitchen scene."
    cleaned = filter_unsupported_claims_verified(text, verified)
    lower = cleaned.lower()
    assert "person and person" not in lower
    assert "man" not in lower and "woman" not in lower
    assert "two people" in lower or "people" in lower


def test_humanize_collapses_person_and_person() -> None:
    text = "A person and person are in a kitchen. Two people are visible in the scene."
    out = sanitize_caption(humanize_caption_style(text))
    lower = out.lower()
    assert "person and person" not in lower
    assert lower.count("people are visible") <= 1


def test_coverage_skips_census_when_plural_already_stated() -> None:
    verified = _verified_two_people()
    text = "Two people are in a kitchen around a white refrigerator."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    assert "people are visible" not in covered.lower()
