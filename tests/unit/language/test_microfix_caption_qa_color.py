"""Micro-fix regressions: caption metadata leak, QA caption append, entity color bleed."""

from __future__ import annotations

from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
    AttributeSet,
)
from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    VerifiedActivity,
    VerifiedEntity,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.prompts.context_caption import build_context_caption
from language.refinement.caption_arbitration import score_caption_candidate


def test_context_caption_never_emits_observed_activity_label() -> None:
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "person_1", "person", 0.2, "middle-center"),
                SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
            ),
            relations=(),
        ),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    activity="riding a bicycle",
                    confidence=0.9,
                    supporting_node_indices=(0, 1),
                    supporting_relation_types=("riding",),
                    rationale="test",
                ),
            ),
            confidence=0.9,
        ),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="trail",
            time_of_day="day",
            weather="clear",
            indoor_outdoor="outdoor",
            social_context="",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=2,
        dominant_objects=("person", "bicycle"),
        spatial_summary="",
    )
    text = build_context_caption(ctx).text.lower()
    assert "observed activity" not in text
    assert "the location is" not in text
    assert "person, and bicycle" not in text
    assert "riding" in text
    assert "bicycle" in text


def test_metadata_caption_rejected_by_arbitration() -> None:
    verified = VerifiedSceneEvidence(
        entities=(
            VerifiedEntity("person_1", 0, "person", 0.9, narrative_safe=True),
            VerifiedEntity("bicycle_1", 1, "bicycle", 0.8, narrative_safe=True),
        ),
        attributes=(),
        relations=(),
        activities=(
            VerifiedActivity(
                activity="riding a bicycle",
                entity_ids=("person_1", "bicycle_1"),
                object_indices=(0, 1),
                confidence=0.9,
                status=ClaimStatus.OBSERVED,
                source="activity",
                supporting_relations=("riding",),
                narrative_safe=True,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            ),
        ),
        scene=VerifiedSceneContext(indoor_outdoor="outdoor", setting="trail"),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        rejected=(),
    )
    bad = "Person, and bicycle. The location is outdoor. Observed activity: riding a bicycle."
    from language.refinement.caption_sanity import sanitize_caption

    cleaned = sanitize_caption(bad)
    lower = cleaned.lower()
    assert "observed activity:" not in lower
    assert "location is outdoor" not in lower
    assert "person, and bicycle" not in lower
    # Arbitration must score the sanitized form — never ship raw metadata prose.
    score = score_caption_candidate(bad, verified)
    scored_l = score.text.lower()
    assert "observed activity:" not in scored_l
    assert "location is outdoor" not in scored_l
    assert not score.rejected or "riding" in scored_l

def test_qa_does_not_append_caption_after_animals_answer() -> None:
    items = (
        EvidenceItem(
            "object", "horse", "is", "horse", 0.9,
            entity_id="horse_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "horse", "is", "horse", 0.88,
            entity_id="horse_2", object_index=1, claim_status="OBSERVED",
        ),
    )
    caption = (
        "A khaki-colored person wearing a red shirt and beige shoes is beside a "
        "brown horse, leading it outdoors across the field."
    )
    packet = AssistantEvidencePacket(
        objects=("horse_1: horse", "horse_2: horse"),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en=caption,
        items=items,
        from_verified=True,
    )
    messy = "2 horses are visible in the scene.\n\n" + caption
    cleaned = VisualEvidenceRetriever()._strip_appended_caption(messy, packet)
    assert "khaki" not in cleaned.lower()
    assert "leading" not in cleaned.lower()
    assert "horse" in cleaned.lower()

    va = VisionAssistant()
    session = VisionAssistantSession(image_key="horse", evidence=packet)
    ans = va.answer(session, "What other animals are visible?")
    assert "khaki" not in ans.lower()
    assert "\n\n" not in ans
    assert "horse" in ans.lower()


def test_sports_ball_beige_refused_as_ground_bleed() -> None:
    items = (
        EvidenceItem(
            "object", "sports ball", "is", "sports ball", 0.9,
            entity_id="sports_ball_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "sports ball", "color", "beige", 0.9,
            entity_id="sports_ball_1", object_index=0, claim_status="OBSERVED",
        ),
    )
    packet = AssistantEvidencePacket(
        objects=("sports_ball_1: sports ball",),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="",
        items=items,
        from_verified=True,
    )
    ans = VisualEvidenceRetriever().try_direct_answer(
        packet, "What color is the sports ball?"
    )
    assert "beige" not in ans.lower()
    assert "can't reliably" in ans.lower()


def test_bicycle_green_refused_as_grass_bleed() -> None:
    items = (
        EvidenceItem(
            "object", "bicycle", "is", "bicycle", 0.9,
            entity_id="bicycle_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "bicycle", "color", "dark green", 0.85,
            entity_id="bicycle_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "person", "is", "person", 0.9,
            entity_id="person_1", object_index=1, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "shirt_color", "olive", 0.8,
            entity_id="person_1", object_index=1, claim_status="OBSERVED",
        ),
    )
    packet = AssistantEvidencePacket(
        objects=("bicycle_1: bicycle", "person_1: person"),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="",
        items=items,
        from_verified=True,
    )
    ans = VisualEvidenceRetriever().try_direct_answer(packet, "What color is the bicycle?")
    assert "green" not in ans.lower()
    assert "can't reliably" in ans.lower()


def test_prompt_excludes_scene_caption() -> None:
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.9,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
    )
    packet = AssistantEvidencePacket(
        objects=("person_1: person",),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="A person is riding a bicycle outdoors.",
        items=items,
        from_verified=True,
    )
    prompt = VisualEvidenceRetriever().retrieve(
        packet, "How many people are visible?"
    ).prompt_block.lower()
    assert "optional caption" not in prompt
    assert "riding a bicycle outdoors" not in prompt
