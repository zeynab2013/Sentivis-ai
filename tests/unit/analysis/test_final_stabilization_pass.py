"""Final stabilization gap tests: VLM bridge, color precedence, arbitration salvage."""

from __future__ import annotations

from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
from analysis.evidence.vlm_activity_bridge import extract_vlm_activity_candidates
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
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    VerifiedEntity,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from language.refinement.caption_arbitration import arbitrate_captions, score_caption_candidate


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    attrs: tuple[Attribute, ...] = (),
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=(), confidence=0.0),
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


def test_vlm_bridge_holding_bat_requires_relation() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.4, "middle-left"),
            SceneNode(1, "bat_1", "baseball bat", 0.1, "middle-right"),
        ),
        relations=(Relation(0, 1, "holding", 0.88),),
    )
    cands = extract_vlm_activity_candidates(
        "A boy is swinging a baseball bat at a white ball.",
        ctx,
    )
    assert any("bat" in a.activity.lower() for a in cands)


def test_vlm_bridge_rejects_near_bike_as_riding() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "bike_1", "bicycle", 0.2, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.5),),
    )
    cands = extract_vlm_activity_candidates(
        "A person is riding a bicycle near a shop.",
        ctx,
    )
    assert not any("riding" in a.activity.lower() for a in cands)


def test_vlm_bridge_accepts_riding_with_relation() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "moto_1", "motorcycle", 0.2, "middle-right"),
        ),
        relations=(Relation(0, 1, "riding", 0.9),),
    )
    cands = extract_vlm_activity_candidates(
        "A man is riding a dirt bike.",
        ctx,
    )
    assert any("motorcycle" in a.activity.lower() for a in cands)


def test_reasoner_color_never_marked_observed() -> None:
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.4, "middle-center"),),
        attrs=(
            Attribute(0, "confidence", "92%"),
            Attribute(0, "shirt_color", "cyan"),
            Attribute(0, "clothing_color", "cyan"),
        ),
    )
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "shirt_color", "brown", 0.90, "scene_reasoner"),
            EvidenceFact("person #1", "clothing_color", "brown", 0.90, "scene_reasoner"),
            EvidenceFact("person #1", "dominant_color", "olive", 0.85, "scene_reasoner"),
        ),
        ranked_subjects=("person #1",),
        environment_keys=("outdoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    verified = build_verified_scene_evidence(ctx, understanding)
    shirts = [a for a in verified.attributes if a.name == "shirt_color"]
    assert shirts and shirts[0].value.lower() == "cyan"
    assert shirts[0].status == ClaimStatus.OBSERVED
    assert shirts[0].source == "pixel_crop"
    # No brown shirt from reasoner.
    assert not any(
        a.name == "shirt_color" and "brown" in a.value.lower() for a in verified.attributes
    )


def test_people_count_property() -> None:
    verified = VerifiedSceneEvidence(
        entities=(
            VerifiedEntity("person_1", 0, "person", 0.9, narrative_safe=True),
            VerifiedEntity("person_2", 1, "person", 0.8, narrative_safe=True),
            VerifiedEntity("bowl_1", 2, "bowl", 0.7, narrative_safe=True),
            VerifiedEntity("person_3", 3, "person", 0.4, narrative_safe=False),
        ),
        attributes=(),
        relations=(),
        activities=(),
        scene=VerifiedSceneContext(indoor_outdoor="indoor", confidence=0.7),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
    )
    assert verified.people_count == 2


def test_arbitration_prefers_salvaged_rich_over_thin_stub() -> None:
    verified = build_verified_scene_evidence(
        _ctx(
            (
                SceneNode(0, "person_1", "person", 0.4, "middle-left"),
                SceneNode(1, "bat_1", "baseball bat", 0.15, "middle-right"),
                SceneNode(2, "ball_1", "sports ball", 0.05, "top-right"),
                SceneNode(3, "person_2", "person", 0.2, "middle-center"),
            ),
            relations=(Relation(0, 1, "holding", 0.9),),
            attrs=(
                Attribute(0, "confidence", "92%"),
                Attribute(0, "shirt_color", "cyan"),
                Attribute(1, "confidence", "80%"),
                Attribute(1, "color", "dark green"),
                Attribute(2, "confidence", "75%"),
                Attribute(2, "color", "cream"),
                Attribute(3, "confidence", "80%"),
            ),
        )
    )
    rich = (
        "A young boy is holding a dark green baseball bat near a cream sports ball. "
        "A catcher is standing behind the boy wearing a red face mask. "
        "Two people are visible."
    )
    thin = "Person, baseball bat, sports ball, and person."
    winner = arbitrate_captions([thin, rich], verified)
    assert "holding" in winner.lower() or "bat" in winner.lower()
    assert "catcher" not in winner.lower()  # unsupported role stripped
    score = score_caption_candidate(rich, verified)
    assert score.rejected is False or "holding" in score.text.lower()


def test_vlm_football_multisignal_without_iou_relation() -> None:
    """VLM 'playing soccer' + ball + people + grass survives without bbox IoU."""
    from analysis.evidence.vlm_activity_bridge import extract_vlm_activity_candidates
    from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
    from core.contracts.analysis import (
        ActivityHints,
        Attribute,
        AttributeSet,
        EnvironmentInfo,
        SceneContext,
        SceneGraph,
        SceneNode,
    )
    from core.contracts.verified_evidence import ActivityEvidenceLevel

    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "p1", "person", 0.2, "middle-left"),
                SceneNode(1, "p2", "person", 0.15, "middle-right"),
                SceneNode(2, "b1", "sports ball", 0.02, "bottom-center"),
            ),
            relations=(),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(1, "confidence", "85%"),
                Attribute(2, "confidence", "80%"),
            )
        ),
        activities=ActivityHints(activities=(), confidence=0.0),
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
        object_count=3,
        dominant_objects=("person", "sports ball"),
        spatial_summary="",
    )
    vlm = (
        "Two girls are playing soccer on a grassy field. "
        "A white ball is sitting on the grass in front of the two girls."
    )
    cands = extract_vlm_activity_candidates(vlm, ctx)
    assert any("football" in a.activity.lower() for a in cands), cands
    football = next(a for a in cands if "football" in a.activity.lower())
    # Shared multi-signal must bind both people + ball, not person_indices[0] alone.
    assert set(football.supporting_node_indices) == {0, 1, 2}
    verified = build_verified_scene_evidence(ctx, vlm_caption=vlm, vlm_confidence=0.9)
    plays = [
        a
        for a in verified.activities
        if "football" in a.activity.lower() or "playing with a ball" in a.activity.lower()
    ]
    assert plays, verified.activities
    assert plays[0].qa_safe is True
    assert plays[0].evidence_level in {
        ActivityEvidenceLevel.CONFIRMED,
        ActivityEvidenceLevel.SUPPORTED,
    }
    assert plays[0].narrative_safe is True


def test_vlm_does_not_invent_football_from_ball_alone() -> None:
    from analysis.evidence.vlm_activity_bridge import extract_vlm_activity_candidates
    from core.contracts.analysis import (
        ActivityHints,
        AttributeSet,
        EnvironmentInfo,
        SceneContext,
        SceneGraph,
        SceneNode,
    )

    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "p1", "person", 0.2, "middle-left"),
                SceneNode(1, "b1", "sports ball", 0.02, "bottom-center"),
            ),
            relations=(),
        ),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.0),
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
        object_count=2,
        dominant_objects=("person", "sports ball"),
        spatial_summary="",
    )
    cands = extract_vlm_activity_candidates(
        "A person stands near a sports ball on the grass.",
        ctx,
    )
    assert not any("football" in a.activity.lower() for a in cands)
    assert not any("playing" in a.activity.lower() for a in cands)
