"""Root-cause regression tests from forensic production traces."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
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
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from core.contracts.verified_evidence import ActivityEvidenceLevel, ClaimStatus


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    attrs: tuple[Attribute, ...] = (),
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=activities, confidence=0.7),
        environment=EnvironmentInfo(
            scene_type="outdoor scene",
            setting="outdoor area",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes[:4]),
        spatial_summary="",
    )


def test_holding_sport_item_is_narrative_safe() -> None:
    """Forensic: CONFIRMED holding existed but narr=False dropped it from caption projection."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.4, "middle-left"),
            SceneNode(1, "bat_1", "baseball bat", 0.1, "middle-right"),
        ),
        relations=(Relation(0, 1, "holding", 0.88),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "80%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    holds = [
        a
        for a in verified.activities
        if "holding" in a.activity.lower() and "bat" in a.activity.lower()
    ]
    assert holds, verified.activities
    assert holds[0].evidence_level == ActivityEvidenceLevel.CONFIRMED
    assert holds[0].narrative_safe is True
    assert holds[0].qa_safe is True


def test_observed_clothing_not_overwritten_by_weaker_reasoner() -> None:
    """Forensic: person_1 cyan shirt OBSERVED was overwritten by brown INFERRED."""
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.4, "middle-center"),),
        attrs=(
            Attribute(0, "confidence", "92%"),
            Attribute(0, "shirt_color", "cyan"),
            Attribute(0, "clothing_color", "cyan"),
            Attribute(0, "pants_color", "charcoal"),
        ),
    )
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "shirt_color", "brown", 0.70, "scene_reasoner"),
            EvidenceFact("person #1", "clothing_color", "brown", 0.70, "scene_reasoner"),
        ),
        ranked_subjects=("person #1",),
        environment_keys=("outdoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.7,
        discarded_count=0,
        contradictions_resolved=0,
    )
    verified = build_verified_scene_evidence(ctx, understanding)
    shirts = [a for a in verified.attributes if a.name == "shirt_color"]
    assert shirts
    assert shirts[0].value.lower() == "cyan"
    assert shirts[0].status == ClaimStatus.OBSERVED


def test_clothing_attrs_never_bind_to_bowl_or_bicycle() -> None:
    """Forensic: bowl_1.pants_color / bicycle_1.pants_color entity bleed."""
    ctx = _ctx(
        (
            SceneNode(0, "bowl_1", "bowl", 0.05, "bottom-left"),
            SceneNode(1, "person_1", "person", 0.2, "middle-center"),
            SceneNode(2, "bicycle_1", "bicycle", 0.15, "middle-right"),
        ),
        attrs=(
            Attribute(0, "confidence", "70%"),
            Attribute(0, "pants_color", "black"),  # wrongly indexed crop leak
            Attribute(0, "clothing_color", "brown"),
            Attribute(1, "confidence", "90%"),
            Attribute(1, "shirt_color", "red"),
            Attribute(1, "clothing_color", "red"),
            Attribute(2, "confidence", "80%"),
            Attribute(2, "pants_color", "light blue"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    for attr in verified.attributes:
        if attr.name in {"pants_color", "shirt_color", "clothing_color"}:
            assert attr.entity_id.startswith("person"), attr
    assert any(
        r.reason.startswith("clothing_attr_on_non_person") for r in verified.rejected
    )
