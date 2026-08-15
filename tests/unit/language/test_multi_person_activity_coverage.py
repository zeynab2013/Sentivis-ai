"""Multi-person verified activity coverage — general regressions."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    Attribute,
    AttributeSet,
    EnvironmentInfo,
    Relation,
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
from language.semantic.natural_caption_service import NaturalCaptionService, _StoryFacts


def _env(setting: str = "kitchen", indoor: str = "indoor") -> EnvironmentInfo:
    return EnvironmentInfo(
        scene_type=setting,
        setting=setting,
        time_of_day="day",
        weather="unknown",
        indoor_outdoor=indoor,
        social_context="",
        crowd_level="few",
        scene_complexity="medium",
        evidence=(f"{setting} scene.",),
    )


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    attrs: tuple[Attribute, ...] = (),
    env: EnvironmentInfo | None = None,
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(
            attributes=attrs
            or tuple(Attribute(i, "confidence", "90%") for i, _ in enumerate(nodes))
        ),
        activities=ActivityHints(activities=activities, confidence=0.85 if activities else 0.0),
        environment=env or _env(),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes),
        spatial_summary="",
    )


def _people_verified(
    activities: tuple[VerifiedActivity, ...],
    *,
    extra_entities: tuple[VerifiedEntity, ...] = (),
    setting: str = "outdoor area",
    indoor: str = "outdoor",
) -> VerifiedSceneEvidence:
    entities = (
        VerifiedEntity(
            entity_id="person_1",
            object_index=0,
            label="person",
            confidence=0.9,
            narrative_safe=True,
        ),
        VerifiedEntity(
            entity_id="person_2",
            object_index=1,
            label="person",
            confidence=0.88,
            narrative_safe=True,
        ),
        *extra_entities,
    )
    return VerifiedSceneEvidence(
        entities=entities,
        attributes=(),
        relations=(),
        activities=activities,
        scene=VerifiedSceneContext(
            indoor_outdoor=indoor,
            setting=setting,
            scene_type=setting,
            confidence=0.8,
            status=ClaimStatus.OBSERVED,
        ),
        ocr_text=(),
        evidence_brief="multi person scene",
        overall_confidence=0.85,
        ranked_entity_ids=tuple(e.entity_id for e in entities),
    )


def test_two_people_different_verified_activities_both_in_caption() -> None:
    """Person A cooking + Person B using phone — both CONFIRMED activities must appear."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-left"),
            SceneNode(1, "person_2", "person", 0.22, "middle-right"),
            SceneNode(2, "oven_1", "oven", 0.12, "middle-center"),
            SceneNode(3, "cell_phone_1", "cell phone", 0.04, "middle-right"),
        ),
        relations=(
            Relation(0, 2, "near", 0.85),
            Relation(1, 3, "holding", 0.88),
            Relation(1, 3, "using", 0.82),
        ),
        activities=(
            ActivityEvidence("cooking", 0.90, (0, 2), ("near",), "cooking"),
            ActivityEvidence(
                "looking at a phone",
                0.88,
                (1, 3),
                ("holding", "using"),
                "looking at a phone",
            ),
        ),
        env=_env("kitchen", "indoor"),
    )
    verified = build_verified_scene_evidence(ctx)
    confirmed = [
        a
        for a in verified.activities
        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
    ]
    understanding = language_understanding_from_verified(verified)
    activity_facts = [f for f in understanding.facts if f.predicate == "activity"]
    assert len(activity_facts) >= 1 or len(confirmed) >= 1

    stub = "Two people are in a kitchen near a dining table."
    covered = ensure_salient_verified_coverage(stub, verified=verified)
    lower = covered.lower()
    if any(
        "cook" in (a.activity or "").lower() or "prepar" in (a.activity or "").lower()
        for a in confirmed
    ):
        assert any(tok in lower for tok in ("cook", "preparing", "food"))
    if any(
        "phone" in (a.activity or "").lower() or "looking" in (a.activity or "").lower()
        for a in confirmed
    ):
        assert any(tok in lower for tok in ("phone", "looking", "using"))


def test_coverage_keeps_second_activity_despite_shared_verb() -> None:
    """Sharing a verb token must not suppress a second distinct activity."""
    verified = _people_verified(
        (
            VerifiedActivity(
                activity="holding a rope",
                entity_ids=("person_1",),
                object_indices=(0,),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=(),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
            VerifiedActivity(
                activity="holding a phone",
                entity_ids=("person_2",),
                object_indices=(1,),
                confidence=0.88,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=(),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        )
    )
    text = "One person is holding a rope outdoors."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    lower = covered.lower()
    assert "rope" in lower
    assert "phone" in lower
    assert "another person" in lower or lower.count("holding") >= 2


def test_secondary_person_clause_uses_distinct_activity() -> None:
    service = NaturalCaptionService.__new__(NaturalCaptionService)
    story = _StoryFacts(
        scene_type="person",
        people=("person #1", "person #2"),
        main="person #1",
        main_label="person",
        main_color="",
        action="cooking",
        primary_interaction="",
        clothing_by_person={},
        objects=("an oven",),
        background_objects=(),
        relations=(),
        place="kitchen",
        weather="",
        time_of_day="day",
        atmosphere="kitchen",
        ocr=(),
        secondary=(),
        omit_reasons=(),
        story_thesis="",
        person_activities=(
            ("person #1", "cooking"),
            ("person #2", "looking at a phone"),
        ),
    )
    clause = service._secondary_person_clause(story, "is cooking")
    assert "phone" in clause.lower() or "looking" in clause.lower()
    assert "farther back" not in clause.lower()


def test_single_person_single_activity_unchanged() -> None:
    verified = VerifiedSceneEvidence(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                object_index=0,
                label="person",
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="bicycle_1",
                object_index=1,
                label="bicycle",
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
        attributes=(),
        relations=(),
        activities=(
            VerifiedActivity(
                activity="riding a bicycle",
                entity_ids=("person_1", "bicycle_1"),
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
        scene=VerifiedSceneContext(
            indoor_outdoor="outdoor",
            setting="street",
            scene_type="street",
            confidence=0.8,
            status=ClaimStatus.OBSERVED,
        ),
        ocr_text=(),
        evidence_brief="person riding",
        overall_confidence=0.9,
        ranked_entity_ids=("person_1", "bicycle_1"),
    )
    text = "A person is riding a bicycle on a street."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    assert "riding" in covered.lower()
    assert covered.lower().count("riding") == 1


def test_uncertain_activity_not_invented_in_coverage() -> None:
    verified = _people_verified(
        (
            VerifiedActivity(
                activity="talking",
                entity_ids=("person_1", "person_2"),
                object_indices=(0, 1),
                confidence=0.5,
                status=ClaimStatus.INFERRED,
                source="activity",
                supporting_relations=(),
                narrative_safe=False,
                qa_safe=False,
                evidence_level=ActivityEvidenceLevel.UNKNOWN,
            ),
        ),
        setting="room",
        indoor="indoor",
    )
    text = "Two people are in a room near a table."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    assert "talking" not in covered.lower()


def test_multi_person_one_verified_activity_no_invention() -> None:
    verified = _people_verified(
        (
            VerifiedActivity(
                activity="leading a horse",
                entity_ids=("person_1", "horse_1"),
                object_indices=(0, 2),
                confidence=0.91,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=("leading",),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
        extra_entities=(
            VerifiedEntity(
                entity_id="horse_1",
                object_index=2,
                label="horse",
                confidence=0.92,
                narrative_safe=True,
            ),
        ),
        setting="field",
        indoor="outdoor",
    )
    text = "A person is leading a horse in a field."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    lower = covered.lower()
    assert "leading" in lower
    assert "cooking" not in lower
    assert "talking" not in lower
    assert "phone" not in lower
