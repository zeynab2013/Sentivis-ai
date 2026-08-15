"""Final-pass QA evidence-flow regressions (color precedence + activity they-doing)."""

from __future__ import annotations

from language.assistant.entity_indexing import (
    IndexedPerson,
    find_person_attribute,
    ordered_people,
    resolve_person_reference,
)
from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem
from language.assistant.vision_assistant import AssistantTurn, VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


def _packet(items: tuple[EvidenceItem, ...], caption: str = "") -> AssistantEvidencePacket:
    return AssistantEvidencePacket(
        objects=tuple(
            f"{i.entity_id}: {i.subject}" for i in items if i.kind == "object"
        ),
        attributes=(),
        relations=(),
        activities=tuple(i.value for i in items if i.kind == "activity"),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en=caption,
        items=items,
        from_verified=True,
    )


def test_observed_shirt_preferred_over_ambiguous_clothing_color() -> None:
    """Khaki/olive aggregate must not beat OBSERVED shirt_color light blue."""
    person = IndexedPerson(
        ordinal=1,
        entity_id="person_1",
        object_index=0,
        label="person",
        confidence=0.9,
        area_ratio=0.2,
    )
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.9,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "shirt_color", "light blue", 0.90,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "clothing_color", "khaki", 0.90,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "color", "cream", 0.90,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
    )
    packet = _packet(items)
    best = find_person_attribute(
        packet, person, predicates=("shirt_color", "clothing_color"), require_reliable=True
    )
    assert best is not None
    assert best.value.lower() == "light blue"
    r = VisualEvidenceRetriever()
    ans = r.try_direct_answer(packet, "What color clothing is the person wearing?")
    assert "light blue" in ans.lower()
    assert "khaki" not in ans.lower()


def test_they_doing_uses_confirmed_riding_with_one_person() -> None:
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "motorcycle", "is", "motorcycle", 0.9,
            entity_id="motorcycle_1", object_index=1, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "activity", "person", "activity", "riding a motorcycle", 0.9,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
            evidence_level="CONFIRMED",
        ),
    )
    packet = _packet(items, caption="A person is riding a motorcycle.")
    r = VisualEvidenceRetriever()
    for q in (
        "What are they doing?",
        "What is the person doing?",
        "What is the person doing in this scene?",
        "What activity is the person performing?",
    ):
        ans = r.try_direct_answer(packet, q)
        assert "riding" in ans.lower(), (q, ans)
        assert "can't determine" not in ans.lower(), (q, ans)


def test_activity_followup_not_rewritten_to_object() -> None:
    """After an object answer, 'What are they doing?' must stay an activity question."""
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "motorcycle", "is", "motorcycle", 0.9,
            entity_id="motorcycle_1", object_index=1, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "activity", "person", "activity", "riding a motorcycle", 0.9,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
            evidence_level="CONFIRMED",
        ),
    )
    packet = _packet(items, caption="A person is riding a motorcycle.")
    va = VisionAssistant()
    session = VisionAssistantSession(image_key="moto", evidence=packet)
    session.turns = [
        AssistantTurn(role="user", text="What objects are visible?"),
        AssistantTurn(role="assistant", text="A motorcycle and a person are visible."),
    ]
    ans = va.answer(session, "What are they doing?")
    assert "riding" in ans.lower()
    assert "can't determine" not in ans.lower()


def test_unindexed_person_prefers_confirmed_activity_actor() -> None:
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.7,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "person", "is", "person", 0.9,
            entity_id="person_2", object_index=1, claim_status="OBSERVED",
        ),
        # Encode area via objects lines used by ordered_people zone/area parser
        EvidenceItem(
            "activity", "person", "activity", "leading a horse", 0.9,
            entity_id="person_2", object_index=1, claim_status="OBSERVED",
            evidence_level="CONFIRMED",
        ),
    )
    packet = AssistantEvidencePacket(
        objects=(
            "person_1: person zone=top-right area=0.02",
            "person_2: person zone=middle-center area=0.20",
        ),
        attributes=(),
        relations=(),
        activities=("leading a horse",),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="",
        items=items,
        from_verified=True,
    )
    people = ordered_people(packet)
    assert len(people) == 2
    resolved = resolve_person_reference("What color clothing is the person wearing?", packet)
    assert resolved is not None
    assert resolved.entity_id == "person_2"


def test_caption_color_does_not_steal_from_other_person() -> None:
    """Multi-person: caption 'navy coat' for person_3 must not answer for person_1."""
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.94,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "object", "person", "is", "person", 0.87,
            entity_id="person_3", object_index=2, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "clothing_color", "olive", 0.94,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "pants_color", "olive", 0.94,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "color", "olive", 0.94,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "dominant_color", "olive", 0.94,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "color", "navy", 0.87,
            entity_id="person_3", object_index=2, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "shirt_color", "royal blue", 0.87,
            entity_id="person_3", object_index=2, claim_status="OBSERVED",
        ),
    )
    packet = AssistantEvidencePacket(
        objects=(
            "person_1: person zone=middle-center area=0.29",
            "person_3: person zone=top-right area=0.01",
        ),
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en="A fourth person, dressed in a navy coat, is beside them.",
        items=items,
        from_verified=True,
    )
    ans = VisualEvidenceRetriever().try_direct_answer(
        packet, "What color clothing is the person wearing?"
    )
    lower = ans.lower()
    assert "navy" not in lower
    # Multi-person unindexed ask must not invent another person's color.
    assert ("which person" in lower) or ("can't reliably" in lower)
    assert "coat" not in lower
    assert "shorts" not in lower


def test_observed_red_shirt_not_overwritten_by_brown_body_color() -> None:
    items = (
        EvidenceItem(
            "object", "person", "is", "person", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "shirt_color", "red", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "clothing_color", "red", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "pants_color", "brown", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "color", "brown", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
        EvidenceItem(
            "attribute", "person", "dominant_color", "brown", 0.93,
            entity_id="person_1", object_index=0, claim_status="OBSERVED",
        ),
    )
    packet = _packet(items, caption="A person wearing a red jersey is riding a motorcycle.")
    ans = VisualEvidenceRetriever().try_direct_answer(
        packet, "What color clothing is the person wearing?"
    )
    assert "red" in ans.lower()
    assert "brown" not in ans.lower()
