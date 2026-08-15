"""Shared multi-person activity ownership + coverage regressions."""

from __future__ import annotations

from analysis.evidence.vlm_activity_bridge import extract_vlm_activity_candidates
from core.contracts.analysis import (
    ActivityHints,
    Attribute,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
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
from language.validation.caption_factuality import filter_unsupported_claims_verified


def _person(eid: str, idx: int, *, area: float = 0.2) -> VerifiedEntity:
    return VerifiedEntity(
        entity_id=eid,
        object_index=idx,
        label="person",
        confidence=0.9,
        bbox=None,
        position_zone="middle-center",
        area_ratio=area,
        narrative_safe=True,
        source="detector",
    )


def _ball(idx: int = 3) -> VerifiedEntity:
    return VerifiedEntity(
        entity_id="sports_ball_1",
        object_index=idx,
        label="sports ball",
        confidence=0.9,
        narrative_safe=True,
    )


def _scene(*entities: VerifiedEntity, activities: tuple[VerifiedActivity, ...]) -> VerifiedSceneEvidence:
    return VerifiedSceneEvidence(
        entities=entities,
        attributes=(),
        relations=(),
        activities=activities,
        scene=VerifiedSceneContext(
            indoor_outdoor="outdoor",
            setting="field",
            scene_type="outdoor",
            confidence=0.8,
        ),
        ocr_text=(),
        evidence_brief="soccer",
        overall_confidence=0.9,
    )


def _shared_football(
    *person_ids: str,
    object_indices: tuple[int, ...] = (0, 1, 3),
) -> VerifiedActivity:
    return VerifiedActivity(
        activity="playing football",
        entity_ids=person_ids,
        object_indices=object_indices,
        confidence=0.9,
        status=ClaimStatus.OBSERVED,
        source="activity",
        supporting_relations=("vlm_multisignal",),
        narrative_safe=True,
        qa_safe=True,
        evidence_level=ActivityEvidenceLevel.CONFIRMED,
    )


def test_soccer_shared_activity_not_one_person() -> None:
    verified = _scene(
        _person("person_1", 0, area=0.40),
        _person("person_2", 1, area=0.28),
        _person("person_3", 2, area=0.02),
        _ball(3),
        activities=(_shared_football("person_1", "person_2", "sports_ball_1"),),
    )
    draft = "A white sports ball rests in the scene."
    covered = ensure_salient_verified_coverage(draft, verified=verified)
    lower = covered.lower()
    assert "one person is playing football" not in lower
    assert "two people are playing football" in lower
    assert "people are visible" in lower or "3 people" in lower or "three people" in lower


def test_two_person_shared_activity_plural() -> None:
    verified = _scene(
        _person("person_1", 0),
        _person("person_2", 1),
        _ball(2),
        activities=(_shared_football("person_1", "person_2", "sports_ball_1", object_indices=(0, 1, 2)),),
    )
    covered = ensure_salient_verified_coverage("A sports ball sits outdoors.", verified=verified)
    assert "two people are playing football" in covered.lower()
    assert "one person is playing" not in covered.lower()


def test_background_people_not_assigned_shared_activity() -> None:
    """4 people in scene, only 2 are activity actors → attribute to 2, not 4."""
    verified = _scene(
        _person("person_1", 0, area=0.35),
        _person("person_2", 1, area=0.30),
        _person("person_3", 2, area=0.05),
        _person("person_4", 3, area=0.04),
        _ball(4),
        activities=(_shared_football("person_1", "person_2", "sports_ball_1", object_indices=(0, 1, 4)),),
    )
    covered = ensure_salient_verified_coverage("A sports ball sits outdoors.", verified=verified)
    lower = covered.lower()
    assert "two people are playing football" in lower
    assert "four people are playing" not in lower
    assert "one person is playing" not in lower


def test_single_person_motorcycle_remains_singular() -> None:
    verified = VerifiedSceneEvidence(
        entities=(
            _person("person_1", 0),
            VerifiedEntity(
                entity_id="motorcycle_1",
                object_index=1,
                label="motorcycle",
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
        attributes=(),
        relations=(),
        activities=(
            VerifiedActivity(
                activity="riding a motorcycle",
                entity_ids=("person_1", "motorcycle_1"),
                object_indices=(0, 1),
                confidence=0.92,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=("riding",),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
        scene=VerifiedSceneContext(indoor_outdoor="outdoor", confidence=0.8),
        ocr_text=(),
        evidence_brief="moto",
        overall_confidence=0.9,
    )
    covered = ensure_salient_verified_coverage(
        "A red motorcycle is visible outdoors.", verified=verified
    )
    lower = covered.lower()
    assert "a person is riding a motorcycle" in lower or "one person is riding" in lower
    assert "two people are riding" not in lower
    assert "people are riding" not in lower


def test_farm_distinct_activities_unchanged() -> None:
    verified = _scene(
        _person("person_1", 0),
        _person("person_2", 1),
        VerifiedEntity(
            entity_id="horse_1",
            object_index=2,
            label="horse",
            confidence=0.9,
            narrative_safe=True,
        ),
        activities=(
            VerifiedActivity(
                activity="leading a horse",
                entity_ids=("person_1", "horse_1"),
                object_indices=(0, 2),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=("leading",),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
            VerifiedActivity(
                activity="holding a rope",
                entity_ids=("person_2",),
                object_indices=(1,),
                confidence=0.88,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=("holding",),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
    )
    covered = ensure_salient_verified_coverage(
        "A brown horse stands outdoors.", verified=verified
    )
    lower = covered.lower()
    assert "leading" in lower and "holding" in lower
    assert "one person is leading" in lower or "a person is leading" in lower
    assert "another person is holding" in lower or "holding a rope" in lower
    assert "two people are leading" not in lower


def test_pipeline_order_soccer_no_one_person_playing() -> None:
    """Reproduce lock order: filter → sanitize → ensure_salient."""
    verified = _scene(
        _person("person_1", 0, area=0.40),
        _person("person_2", 1, area=0.28),
        _person("person_3", 2, area=0.02),
        _ball(3),
        activities=(_shared_football("person_1", "person_2", "sports_ball_1"),),
    )
    draft = (
        "Two girls are playing soccer on a grassy field. "
        "A white sports ball rests in the scene."
    )
    filtered = filter_unsupported_claims_verified(draft, verified) or draft
    cleaned = sanitize_caption(filtered)
    final = ensure_salient_verified_coverage(cleaned, verified=verified)
    lower = final.lower()
    assert "one person is playing football" not in lower
    assert "playing football" in lower or "playing soccer" in lower
    # Shared agency preserved somehow (plural people + playing, or two people are playing).
    assert (
        "two people are playing football" in lower
        or ("people" in lower and "playing" in lower and "one person is playing" not in lower)
    )


def test_vlm_bridge_binds_two_largest_people_not_all() -> None:
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "p1", "person", 0.40, "middle-left"),
                SceneNode(1, "p2", "person", 0.28, "middle-right"),
                SceneNode(2, "p3", "person", 0.02, "background"),
                SceneNode(3, "p4", "person", 0.015, "background"),
                SceneNode(4, "b1", "sports ball", 0.02, "bottom-center"),
            ),
            relations=(),
        ),
        attributes=AttributeSet(
            attributes=tuple(Attribute(i, "confidence", "90%") for i in range(5))
        ),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="football field",
            setting="field",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="general",
            crowd_level="few",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=5,
        dominant_objects=("person", "sports ball"),
        spatial_summary="",
    )
    vlm = (
        "Two girls are playing soccer on a grassy field. "
        "A white ball sits in front of the two girls. Other people stand farther back."
    )
    cands = extract_vlm_activity_candidates(vlm, ctx)
    football = [c for c in cands if "football" in c.activity.lower()]
    assert football, cands
    nodes = football[0].supporting_node_indices
    person_nodes = [i for i in nodes if i in {0, 1, 2, 3}]
    assert 4 in nodes  # ball
    assert set(person_nodes) == {0, 1}  # two largest, not tiny background
    assert 2 not in person_nodes and 3 not in person_nodes
