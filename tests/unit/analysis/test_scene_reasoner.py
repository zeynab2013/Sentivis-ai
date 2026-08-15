"""SceneReasoner unit tests."""

from analysis.pose.pose_estimator import PoseEstimate
from analysis.scene_reasoner.scene_reasoner import SceneReasoner
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
from core.contracts.language import RawCaption, VisualObservations


def test_scene_reasoner_merges_and_discards_low_confidence() -> None:
    context = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "obj-0", "person", 0.2, "middle-center"),
                SceneNode(1, "obj-1", "umbrella", 0.05, "middle-right"),
            ),
            relations=(Relation(0, 1, "holding", 0.8),),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(0, "segmentation", "mask"),
                Attribute(0, "visibility", "high"),
                Attribute(0, "occlusion", "none"),
                Attribute(0, "shirt_color", "red"),
                Attribute(0, "pants_color", "blue"),
                Attribute(0, "clothing_color", "red"),
                Attribute(0, "clothing_palette", "red, blue"),
                Attribute(0, "clothing_type", "hoodie"),
                Attribute(0, "hoodie", "likely"),
                Attribute(1, "confidence", "30%"),
                Attribute(1, "segmentation", "bbox"),
                Attribute(1, "dominant_color", "black"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    activity="holding umbrella",
                    confidence=0.8,
                    supporting_node_indices=(0, 1),
                    supporting_relation_types=("holding",),
                    rationale="person near umbrella",
                ),
            ),
            confidence=0.8,
        ),
        environment=EnvironmentInfo(
            scene_type="street",
            setting="street",
            time_of_day="afternoon",
            weather="rainy",
            indoor_outdoor="outdoor",
            social_context="none",
            crowd_level="sparse",
            scene_complexity="medium",
            evidence=("umbrella",),
        ),
        object_count=2,
        dominant_objects=("person", "umbrella"),
        spatial_summary="Person holding umbrella.",
    )
    poses = (
        PoseEstimate(0, "standing", "holding umbrella", 0.8, "pose_estimator", 1.0),
    )
    understanding = SceneReasoner().reason(context, poses=poses)
    assert understanding.facts
    assert understanding.evidence_brief
    assert "HIGH-CONFIDENCE EVIDENCE PACKAGE" in understanding.evidence_brief
    assert "People" in understanding.evidence_brief or "shirt_color" in understanding.evidence_brief
    assert "vlm_observation" not in understanding.evidence_brief
    assert any(fact.predicate == "shirt_color" and fact.value == "red" for fact in understanding.facts)
    assert any(fact.predicate == "holding" for fact in understanding.facts)
    # Low-confidence umbrella object attributes should be discarded.
    assert not any(fact.subject == "umbrella" and fact.predicate == "dominant_color" for fact in understanding.facts)


def test_scene_reasoner_recovers_fire_from_vlm() -> None:
    context = SceneContext(
        graph=SceneGraph(
            nodes=(SceneNode(0, "obj-0", "person", 0.15, "middle-center"),),
            relations=(),
        ),
        attributes=AttributeSet(attributes=(Attribute(0, "confidence", "88%"),)),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="field",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="none",
            crowd_level="sparse",
            scene_complexity="medium",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="Person outdoors.",
    )
    observations = VisualObservations(
        observations=("campfire", "flames", "smoke"),
        object_attributes=(),
        candidate_descriptions=(),
        confidence=0.82,
        raw_caption=RawCaption(
            text="A person stands near a campfire with visible flames and smoke.",
            source="vlm",
            confidence=0.82,
        ),
    )
    understanding = SceneReasoner().reason(context, observations=observations)
    assert any(f.subject == "fire" and f.predicate == "is" for f in understanding.facts)
    assert any(f.predicate == "hazard" and f.value == "fire" for f in understanding.facts)
    assert any(f.subject == "smoke" and f.predicate == "is" for f in understanding.facts)
