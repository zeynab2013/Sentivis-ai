"""Preserve activity/relationship actor quantities vs scene census in clamp."""

from __future__ import annotations

from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    VerifiedActivity,
    VerifiedEntity,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from language.refinement.caption_coverage import ensure_salient_verified_coverage
from language.refinement.caption_sanity import sanitize_caption
from language.validation.caption_factuality import (
    clamp_caption_object_counts,
    filter_unsupported_claims_verified,
)


def _people(*ids: str) -> tuple[VerifiedEntity, ...]:
    return tuple(
        VerifiedEntity(
            entity_id=eid,
            object_index=i,
            label="person",
            confidence=0.9,
            narrative_safe=True,
            area_ratio=0.2 if i < 2 else 0.05,
        )
        for i, eid in enumerate(ids)
    )


def _verified(
    people: tuple[VerifiedEntity, ...],
    activities: tuple[VerifiedActivity, ...] = (),
    extra: tuple[VerifiedEntity, ...] = (),
) -> VerifiedSceneEvidence:
    return VerifiedSceneEvidence(
        entities=people + extra,
        attributes=(),
        relations=(),
        activities=activities,
        scene=VerifiedSceneContext(indoor_outdoor="outdoor", confidence=0.8),
        ocr_text=(),
        evidence_brief="test",
        overall_confidence=0.9,
    )


def _act(
    name: str,
    *person_ids: str,
    obj_indices: tuple[int, ...] | None = None,
) -> VerifiedActivity:
    indices = obj_indices
    if indices is None:
        indices = tuple(range(len(person_ids)))
    return VerifiedActivity(
        activity=name,
        entity_ids=person_ids,
        object_indices=indices,
        confidence=0.9,
        status=ClaimStatus.OBSERVED,
        source="activity",
        supporting_relations=(),
        narrative_safe=True,
        qa_safe=True,
        evidence_level=ActivityEvidenceLevel.CONFIRMED,
    )


def test_clamp_preserves_two_football_actors_among_four_people() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3", "person_4"),
        (_act("playing football", "person_1", "person_3"),),
    )
    text = "Four people are visible. Two people are playing football."
    out = clamp_caption_object_counts(text, verified=verified)
    lower = out.lower()
    assert "two people are playing football" in lower
    assert "four people are playing football" not in lower
    assert "4 people are playing football" not in lower
    assert "people are visible" in lower


def test_clamp_preserves_two_tennis_actors_among_three() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3"),
        (_act("playing tennis", "person_1", "person_2"),),
    )
    out = clamp_caption_object_counts(
        "Three people are visible. Two people are playing tennis.",
        verified=verified,
    ).lower()
    assert "two people are playing tennis" in out
    assert "three people are playing tennis" not in out


def test_clamp_preserves_one_cooking_actor_among_five() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3", "person_4", "person_5"),
        (_act("preparing food", "person_1"),),
    )
    out = clamp_caption_object_counts(
        "Five people are in the kitchen. One person is preparing food.",
        verified=verified,
    ).lower()
    assert "one person is preparing food" in out
    assert "five people are preparing food" not in out


def test_clamp_preserves_shared_object_holders() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3", "person_4"),
        (_act("holding a banner", "person_1", "person_2"),),
    )
    out = clamp_caption_object_counts(
        "Four people are visible. Two people are holding a banner.",
        verified=verified,
    ).lower()
    assert "two people are holding a banner" in out
    assert "four people are holding" not in out


def test_clamp_preserves_singular_rider() -> None:
    people = _people("person_1")
    moto = VerifiedEntity(
        entity_id="motorcycle_1",
        object_index=1,
        label="motorcycle",
        confidence=0.9,
        narrative_safe=True,
    )
    verified = _verified(
        people,
        (_act("riding a motorcycle", "person_1", "motorcycle_1", obj_indices=(0, 1)),),
        extra=(moto,),
    )
    out = clamp_caption_object_counts(
        "A person is riding a motorcycle.",
        verified=verified,
    ).lower()
    assert "a person is riding a motorcycle" in out
    assert "people are riding" not in out


def test_clamp_one_rider_among_four_people() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3", "person_4"),
        (_act("riding a motorcycle", "person_1"),),
    )
    out = clamp_caption_object_counts(
        "Four people are visible. One person is riding a motorcycle.",
        verified=verified,
    ).lower()
    assert "one person is riding a motorcycle" in out
    assert "four people are riding" not in out


def test_clamp_preserves_two_of_two_shared_actors() -> None:
    verified = _verified(
        _people("person_1", "person_2"),
        (_act("playing football", "person_1", "person_2"),),
    )
    out = clamp_caption_object_counts(
        "Two people are playing football.",
        verified=verified,
    ).lower()
    assert "two people are playing football" in out


def test_clamp_preserves_distinct_activity_ownership() -> None:
    verified = _verified(
        _people("person_1", "person_2", "person_3"),
        (
            _act("riding a bicycle", "person_1"),
            _act("holding a bag", "person_2"),
        ),
    )
    text = (
        "Three people are visible. One person is riding a bicycle. "
        "Another person is holding a bag."
    )
    out = clamp_caption_object_counts(text, verified=verified).lower()
    assert "one person is riding a bicycle" in out or "a person is riding a bicycle" in out
    assert "holding a bag" in out
    assert "three people are riding" not in out
    assert "three people are holding" not in out


def test_clamp_farm_distinct_activities_unchanged() -> None:
    horse = VerifiedEntity(
        entity_id="horse_1",
        object_index=2,
        label="horse",
        confidence=0.9,
        narrative_safe=True,
    )
    verified = _verified(
        _people("person_1", "person_2"),
        (
            _act("leading a horse", "person_1", "horse_1", obj_indices=(0, 2)),
            _act("holding a rope", "person_2"),
        ),
        extra=(horse,),
    )
    text = "One person is leading a horse. Another person is holding a rope. Two people are visible."
    out = clamp_caption_object_counts(text, verified=verified).lower()
    assert "leading a horse" in out
    assert "holding a rope" in out
    assert "two people are leading" not in out
    assert "two people are holding a rope" not in out


def test_clamp_still_normalizes_pure_census() -> None:
    verified = _verified(_people("person_1", "person_2", "person_3", "person_4"))
    out = clamp_caption_object_counts(
        "Two people are visible in the scene.",
        verified=verified,
    ).lower()
    assert "4 people" in out or "four people" in out
    assert "two people are visible" not in out


def test_pipeline_order_soccer_actor_count_survives_clamp() -> None:
    """ensure_salient → clamp must keep Two people playing, not Four."""
    ball = VerifiedEntity(
        entity_id="sports_ball_1",
        object_index=4,
        label="sports ball",
        confidence=0.9,
        narrative_safe=True,
    )
    verified = _verified(
        _people("person_1", "person_2", "person_3", "person_4"),
        (
            _act(
                "playing football",
                "person_1",
                "person_3",
                "sports_ball_1",
                obj_indices=(0, 2, 4),
            ),
        ),
        extra=(ball,),
    )
    draft = "A white sports ball rests in the scene."
    filtered = filter_unsupported_claims_verified(draft, verified) or draft
    cleaned = sanitize_caption(filtered)
    covered = ensure_salient_verified_coverage(cleaned, verified=verified)
    final = clamp_caption_object_counts(covered, verified=verified)
    lower = final.lower()
    assert "two people are playing football" in lower
    assert "four people are playing football" not in lower
    assert "4 people are playing football" not in lower
    assert "one person is playing football" not in lower
