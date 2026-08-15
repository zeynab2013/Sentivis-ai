"""Local patch regressions: caption claim dedupe + activity coverage matching."""

from __future__ import annotations

import time

from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import ActivityEvidence, ActivityHints, AttributeSet
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    VerifiedActivity,
    VerifiedEntity,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.refinement.caption_coverage import ensure_salient_verified_coverage
from language.refinement.caption_sanity import dedupe_semantic_facts, sanitize_caption


def test_duplicate_holding_rope_claim_is_merged() -> None:
    text = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors. "
        "A person is holding a rope."
    )
    cleaned = dedupe_semantic_facts(text).lower()
    assert "holds a rope while leading" in cleaned or "leading a brown horse" in cleaned
    # Restated bare holding claim must not remain as a second sentence.
    assert cleaned.count("holding a rope") + cleaned.count("holds a rope") == 1
    assert "leading" in cleaned
    assert "horse" in cleaned


def test_sanitize_also_drops_restated_rope_hold() -> None:
    text = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors. "
        "A person is holding a rope. Another person stands farther back."
    )
    cleaned = sanitize_caption(text).lower()
    assert "farther back" in cleaned
    assert cleaned.count("holding a rope") + cleaned.count("holds a rope") == 1


def test_activity_coverage_leading_matches_brown_horse_modifier() -> None:
    analysis_config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="obj-person",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(10, 10, 100, 200),
                class_id=0,
                detected_at=now,
            ),
            Detection(
                object_id="obj-horse",
                label="horse",
                confidence=0.9,
                bounding_box=BoundingBox(120, 10, 220, 200),
                class_id=17,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    graph = SceneGraphBuilder(analysis_config).build(detections, ())
    activities = ActivityHints(
        activities=(
            ActivityEvidence(
                activity="leading a horse",
                confidence=0.9,
                supporting_node_indices=(0, 1),
                supporting_relation_types=("leading",),
                rationale="Verified leading.",
            ),
            ActivityEvidence(
                activity="holding a rope",
                confidence=0.85,
                supporting_node_indices=(0,),
                supporting_relation_types=("holding",),
                rationale="Verified holding.",
            ),
        ),
        confidence=0.9,
    )
    context = ContextBuilder(analysis_config).build(graph, AttributeSet(attributes=()), activities)
    caption = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors."
    )
    report = CaptionQualityEvaluator().evaluate(caption, context)
    assert report.activity_coverage is not None
    assert report.activity_coverage == 1.0


def test_activity_coverage_holds_matches_holding() -> None:
    analysis_config = load_analysis_config()
    now = time.time()
    detections = DetectionResult(
        detections=(
            Detection(
                object_id="obj-person",
                label="person",
                confidence=0.9,
                bounding_box=BoundingBox(10, 10, 100, 200),
                class_id=0,
                detected_at=now,
            ),
        ),
        image_width=640,
        image_height=480,
        inference_timestamp=now,
    )
    graph = SceneGraphBuilder(analysis_config).build(detections, ())
    activities = ActivityHints(
        activities=(
            ActivityEvidence(
                activity="holding a rope",
                confidence=0.9,
                supporting_node_indices=(0,),
                supporting_relation_types=("holding",),
                rationale="Verified holding.",
            ),
        ),
        confidence=0.9,
    )
    context = ContextBuilder(analysis_config).build(graph, AttributeSet(attributes=()), activities)
    report = CaptionQualityEvaluator().evaluate("A person holds a rope.", context)
    assert report.activity_coverage == 1.0


def test_unique_verified_claims_remain_after_dedupe() -> None:
    text = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors. "
        "Another person stands farther back in the field."
    )
    cleaned = dedupe_semantic_facts(text).lower()
    assert "leading" in cleaned and "horse" in cleaned
    assert "holds a rope" in cleaned or "holding a rope" in cleaned
    assert "farther back" in cleaned


def _farm_verified(
    *activities: str,
) -> VerifiedSceneEvidence:
    acts = tuple(
        VerifiedActivity(
            activity=name,
            entity_ids=("person_1",),
            object_indices=(0,),
            confidence=0.9,
            status=ClaimStatus.OBSERVED,
            source="activity",
            supporting_relations=(),
            narrative_safe=True,
            qa_safe=True,
            evidence_level=ActivityEvidenceLevel.CONFIRMED,
        )
        for name in activities
    )
    return VerifiedSceneEvidence(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                object_index=0,
                label="person",
                confidence=0.9,
                bbox=None,
                position_zone="middle-center",
                area_ratio=0.2,
                narrative_safe=True,
                source="detector",
            ),
            VerifiedEntity(
                entity_id="horse_1",
                object_index=1,
                label="horse",
                confidence=0.9,
                bbox=None,
                position_zone="middle-center",
                area_ratio=0.25,
                narrative_safe=True,
                source="detector",
            ),
        ),
        attributes=(),
        relations=(),
        activities=acts,
        scene=VerifiedSceneContext(
            indoor_outdoor="outdoor",
            setting="outdoors",
            scene_type="outdoor",
            time_of_day="day",
            weather="unknown",
            crowd_level="",
            confidence=0.8,
        ),
        ocr_text=(),
        evidence_brief="farm",
        overall_confidence=0.9,
    )


def test_sanitize_then_coverage_does_not_reinject_holding_rope() -> None:
    """Real lock order: sanitize/dedupe then ensure_salient must not re-append rope hold."""
    draft = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors. "
        "A fire is burning in the foreground. Farther back, another horse is in the field. "
        "A person is holding a rope."
    )
    verified = _farm_verified("holding a rope", "leading a horse")
    after_sanitize = sanitize_caption(draft)
    assert "a person is holding a rope." not in after_sanitize.lower()
    covered = ensure_salient_verified_coverage(after_sanitize, verified=verified)
    lower = covered.lower()
    assert "a person is holding a rope." not in lower
    assert lower.count("holding a rope") + lower.count("holds a rope") == 1
    assert "leading" in lower and "horse" in lower
    assert "fire" in lower


def test_coverage_recognizes_holds_and_brown_horse_modifiers() -> None:
    verified = _farm_verified("holding a rope", "leading a horse")
    text = (
        "A person wearing black clothing holds a rope while leading a brown horse outdoors."
    )
    covered = ensure_salient_verified_coverage(text, verified=verified)
    lower = covered.lower()
    assert "a person is holding a rope." not in lower
    assert "a person is leading a horse." not in lower
    assert "holds a rope" in lower
    assert "leading a brown horse" in lower


def test_coverage_still_appends_genuinely_missing_activity() -> None:
    verified = _farm_verified("holding a rope", "using a laptop")
    text = "A person wearing black clothing holds a rope while leading a brown horse outdoors."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    lower = covered.lower()
    assert "holds a rope" in lower or "holding a rope" in lower
    assert "laptop" in lower
    # Rope claim expressed once; missing laptop still covered.
    assert lower.count("holding a rope") + lower.count("holds a rope") == 1


def test_coverage_keeps_distinct_second_activity_with_shared_verb() -> None:
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
                entity_id="person_2",
                object_index=1,
                label="person",
                confidence=0.88,
                narrative_safe=True,
            ),
        ),
        attributes=(),
        relations=(),
        activities=(
            VerifiedActivity(
                activity="holding a rope",
                entity_ids=("person_1",),
                object_indices=(0,),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
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
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
        scene=VerifiedSceneContext(indoor_outdoor="outdoor", confidence=0.8),
        ocr_text=(),
        evidence_brief="two people",
        overall_confidence=0.9,
    )
    text = "One person is holding a rope outdoors."
    covered = ensure_salient_verified_coverage(text, verified=verified)
    lower = covered.lower()
    assert "rope" in lower
    assert "phone" in lower
