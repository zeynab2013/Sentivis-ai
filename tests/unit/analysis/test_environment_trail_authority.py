"""Environment authority: preserve trail when scene evidence supports it."""

from __future__ import annotations

from analysis.context.context_builder import ContextBuilder
from analysis.evidence.verified_evidence_builder import (
    _calibrate_scene_label,
    build_verified_scene_evidence,
)
from core.config.loader import load_analysis_config
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
from core.contracts.verified_evidence import VerifiedActivity, VerifiedEntity, ClaimStatus


def _entities(*labels: str) -> list[VerifiedEntity]:
    out: list[VerifiedEntity] = []
    for i, lab in enumerate(labels):
        out.append(
            VerifiedEntity(
                entity_id=f"{lab.replace(' ', '_')}_{i+1}",
                object_index=i,
                label=lab,
                confidence=0.9,
                narrative_safe=True,
            )
        )
    return out


def test_bicycle_alone_does_not_force_trail() -> None:
    scene_type, setting, _ = _calibrate_scene_label(
        "outdoor scene",
        "outdoor area",
        0.7,
        entities=_entities("person", "bicycle"),
        activities=[],
        evidence_blob="a person is riding a bicycle outdoors",
        indoor_outdoor="outdoor",
    )
    assert "trail" not in setting.lower()
    assert setting in {"outdoor area", "outdoor scene", "natural environment"} or "outdoor" in setting


def test_explicit_trail_scene_evidence_preserved() -> None:
    acts = [
        VerifiedActivity(
            activity="riding a bicycle",
            entity_ids=("person_1", "bicycle_1"),
            object_indices=(0, 1),
            confidence=0.9,
            status=ClaimStatus.OBSERVED,
            source="relations",
            narrative_safe=True,
            qa_safe=True,
        )
    ]
    scene_type, setting, conf = _calibrate_scene_label(
        "natural environment",
        "outdoor area",
        0.7,
        entities=_entities("person", "bicycle", "tree"),
        activities=acts,
        evidence_blob=(
            "He is riding the bicycle along a dirt trail. "
            "This suggests an outdoor setting on a trail."
        ),
        indoor_outdoor="outdoor",
    )
    assert setting == "outdoor trail"
    assert "natural" in scene_type or scene_type == "outdoor scene"
    assert conf >= 0.68


def test_horse_alone_still_not_farm() -> None:
    scene_type, setting, _ = _calibrate_scene_label(
        "farm",
        "farm pasture",
        0.8,
        entities=_entities("person", "horse"),
        activities=[],
        evidence_blob="a person is with a brown horse outdoors",
        indoor_outdoor="outdoor",
    )
    assert "farm" not in scene_type.lower()
    assert "farm" not in setting.lower()


def test_bus_still_not_highway() -> None:
    scene_type, setting, _ = _calibrate_scene_label(
        "highway",
        "highway",
        0.8,
        entities=_entities("bus"),
        activities=[],
        evidence_blob="a bus is visible",
        indoor_outdoor="outdoor",
    )
    assert "highway" not in scene_type.lower()
    assert "highway" not in setting.lower()


def test_context_builder_bicycle_nature_not_farm_or_trail() -> None:
    builder = ContextBuilder(load_analysis_config())
    scene_type, setting = builder._specific_outdoor_setting(
        {"person", "bicycle", "tree", "sky"},
        {"riding a bicycle"},
    )
    assert "farm" not in scene_type.lower()
    assert "trail" not in setting.lower()  # trail requires scene-text evidence later


def test_verified_evidence_uses_vlm_trail_caption() -> None:
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "person_1", "person", 0.25, "middle-center"),
                SceneNode(1, "bicycle_1", "bicycle", 0.2, "middle-center"),
            ),
            relations=(Relation(0, 1, "riding", 0.9),),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(1, "confidence", "88%"),
            )
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    "riding a bicycle",
                    0.9,
                    (0, 1),
                    ("riding",),
                    "riding",
                ),
            ),
            confidence=0.9,
        ),
        environment=EnvironmentInfo(
            scene_type="natural environment",
            setting="outdoor area",
            time_of_day="daytime",
            weather="clear",
            indoor_outdoor="outdoor",
            social_context="",
            crowd_level="single person",
            scene_complexity="low",
            evidence=("Outdoor object labels present.",),
        ),
        object_count=2,
        dominant_objects=("person", "bicycle"),
        spatial_summary="",
    )
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact(
                "vlm",
                "observation",
                "A person riding a bicycle along a dirt trail outdoors.",
                0.8,
                "vlm",
            ),
        ),
        ranked_subjects=("person #1", "bicycle #1"),
        environment_keys=("outdoor", "outdoor area"),
        activity_keys=("riding a bicycle",),
        ocr_text=(),
        evidence_brief="person riding bicycle on dirt trail",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    verified = build_verified_scene_evidence(
        ctx,
        understanding,
        vlm_caption="A person riding a bicycle along a dirt trail outdoors.",
    )
    assert verified.scene.setting == "outdoor trail"
    assert "farm" not in (verified.scene.setting or "").lower()
