"""Regression tests for final emergency stabilization restores."""

from __future__ import annotations

from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    VerifiedActivity,
    VerifiedEntity,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.refinement.caption_coverage import ensure_salient_verified_coverage
from language.validation.caption_factuality import filter_unsupported_claims_verified


def _scene() -> VerifiedSceneContext:
    return VerifiedSceneContext(
        indoor_outdoor="outdoor",
        setting="outdoor area",
        scene_type="outdoor scene",
        confidence=0.8,
    )


def _verified(
    *,
    entities: tuple[VerifiedEntity, ...],
    activities: tuple[VerifiedActivity, ...] = (),
) -> VerifiedSceneEvidence:
    return VerifiedSceneEvidence(
        entities=entities,
        attributes=(),
        relations=(),
        activities=activities,
        scene=_scene(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.85,
        ranked_entity_ids=tuple(e.entity_id for e in entities),
    )


def test_confirmed_riding_injected_into_thin_caption() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="bicycle_1",
                label="bicycle",
                object_index=1,
                confidence=0.88,
                narrative_safe=True,
            ),
        ),
        activities=(
            VerifiedActivity(
                activity="riding a bicycle",
                entity_ids=("person_1", "bicycle_1"),
                object_indices=(0, 1),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
    )
    thin = "They are dressed in light blue pants. Another person stands farther back in the scene."
    covered = ensure_salient_verified_coverage(thin, verified=verified)
    assert "riding" in covered.lower()
    assert "bicycle" in covered.lower()


def test_two_people_count_injected_when_missing() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="person_2",
                label="person",
                object_index=1,
                confidence=0.8,
                narrative_safe=True,
            ),
        ),
    )
    thin = "A brown bowl sits near the sink."
    covered = ensure_salient_verified_coverage(thin, verified=verified)
    assert "two people" in covered.lower()


def test_riding_removes_next_to_motorcycle_contradiction() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="motorcycle_1",
                label="motorcycle",
                object_index=1,
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
        activities=(
            VerifiedActivity(
                activity="riding a motorcycle",
                entity_ids=("person_1", "motorcycle_1"),
                object_indices=(0, 1),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
    )
    text = (
        "A single individual wearing a red jersey and brown pants "
        "is positioned next to a brown motorcycle. A person is riding a motorcycle."
    )
    cleaned = ensure_salient_verified_coverage(text, verified=verified)
    assert "riding" in cleaned.lower()
    assert "positioned next to" not in cleaned.lower()
    assert "next to a brown motorcycle" not in cleaned.lower()


def test_riding_removes_positioned_to_the_left_contradiction() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="motorcycle_1",
                label="motorcycle",
                object_index=1,
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
        activities=(
            VerifiedActivity(
                activity="riding a motorcycle",
                entity_ids=("person_1", "motorcycle_1"),
                object_indices=(0, 1),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
    )
    text = (
        "A single individual wearing a red jersey and brown pants "
        "is positioned to the left of a brown motorcycle. A person is riding a motorcycle."
    )
    cleaned = ensure_salient_verified_coverage(text, verified=verified)
    assert "riding" in cleaned.lower()
    assert "to the left of" not in cleaned.lower()
    assert "positioned to the left" not in cleaned.lower()


def test_unsupported_lying_pose_removed() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
    )
    text = "In a kitchen, a person wearing brown clothing is lying. A sink is nearby."
    cleaned = ensure_salient_verified_coverage(text, verified=verified)
    assert "lying" not in cleaned.lower()
    assert "kitchen" in cleaned.lower() or "sink" in cleaned.lower()


def test_gender_sentence_rewritten_not_dropped() -> None:
    verified = _verified(
        entities=(
            VerifiedEntity(
                entity_id="person_1",
                label="person",
                object_index=0,
                confidence=0.9,
                narrative_safe=True,
            ),
            VerifiedEntity(
                entity_id="motorcycle_1",
                label="motorcycle",
                object_index=1,
                confidence=0.9,
                narrative_safe=True,
            ),
        ),
        activities=(
            VerifiedActivity(
                activity="riding a motorcycle",
                entity_ids=("person_1",),
                object_indices=(0, 1),
                confidence=0.92,
                status=ClaimStatus.OBSERVED,
                source="activity",
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
    )
    text = "The man is riding a motorcycle."
    filtered = filter_unsupported_claims_verified(text, verified)
    assert "riding" in filtered.lower()
    assert "motorcycle" in filtered.lower()
    assert "man" not in filtered.lower()


def test_sports_ball_color_not_from_person_clothing() -> None:
    packet = AssistantEvidencePacket(
        objects=(
            "person_1:person (zone=center, area=0.30, conf=0.90, narrative_safe=True)",
            "sports_ball_1:sports ball (zone=bottom, area=0.02, conf=0.80, narrative_safe=True)",
        ),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="",
        items=(
            EvidenceItem(
                kind="object",
                subject="person",
                predicate="detected",
                value="yes",
                confidence=0.9,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="object",
                subject="sports ball",
                predicate="detected",
                value="yes",
                confidence=0.8,
                object_index=1,
                entity_id="sports_ball_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="attribute",
                subject="person",
                predicate="clothing_color",
                value="olive",
                confidence=0.85,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="attribute",
                subject="sports ball",
                predicate="dominant_color",
                value="olive",
                confidence=0.70,
                object_index=1,
                entity_id="sports_ball_1",
                claim_status="OBSERVED",
            ),
        ),
    )
    retriever = VisualEvidenceRetriever()
    answer = retriever._color_of_object_answer(packet, "What color is the sports ball?")
    assert "olive" not in answer.lower()
    assert "reliably" in answer.lower() or "can't" in answer.lower() or "cannot" in answer.lower()


def test_pants_color_separate_from_shirt() -> None:
    packet = AssistantEvidencePacket(
        objects=("person_1:person (zone=center, area=0.40, conf=0.92, narrative_safe=True)",),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="",
        items=(
            EvidenceItem(
                kind="object",
                subject="person",
                predicate="detected",
                value="yes",
                confidence=0.92,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="attribute",
                subject="person",
                predicate="shirt_color",
                value="red",
                confidence=0.88,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="attribute",
                subject="person",
                predicate="pants_color",
                value="brown",
                confidence=0.84,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
            EvidenceItem(
                kind="attribute",
                subject="person",
                predicate="clothing_color",
                value="red",
                confidence=0.88,
                object_index=0,
                entity_id="person_1",
                claim_status="OBSERVED",
            ),
        ),
    )
    retriever = VisualEvidenceRetriever()
    pants = retriever._clothing_answer(packet, "What color pants is the person wearing?")
    shirt = retriever._clothing_answer(packet, "What color is the person's shirt?")
    assert "brown" in pants.lower()
    assert "red" in shirt.lower()
