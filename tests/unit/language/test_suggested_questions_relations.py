"""Suggested questions should explore uncovered relationships."""

from __future__ import annotations

from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem
from language.assistant.suggested_questions import generate_suggested_questions


def _packet() -> AssistantEvidencePacket:
    items = (
        EvidenceItem("object", "person", "is", "person", 0.9),
        EvidenceItem("object", "phone", "is", "phone", 0.85),
        EvidenceItem("object", "chair", "is", "chair", 0.8),
        EvidenceItem("relation", "person", "holding", "phone", 0.82),
        EvidenceItem("relation", "person", "sitting_on", "chair", 0.78),
        EvidenceItem("attribute", "person", "shirt_color", "blue", 0.75),
    )
    return AssistantEvidencePacket(
        objects=("person", "phone", "chair"),
        attributes=("person.shirt_color=blue",),
        relations=("person.holding=phone", "person.sitting_on=chair"),
        activities=(),
        environment=("indoor_outdoor=indoor", "setting=room"),
        ocr=(),
        evidence_brief="person holding phone; sitting_on chair; shirt_color=blue",
        canonical_caption_en="A person is sitting indoors. A chair is nearby.",
        items=items,
    )


def test_suggested_questions_include_relation_not_in_caption() -> None:
    questions = generate_suggested_questions(_packet(), language="en", limit=3)
    assert questions, "expected at least one suggested question"
    joined = " ".join(q.lower() for q in questions)
    # Caption already covers sitting/chair — holding/color should be preferred.
    assert "how many people" not in joined
    assert any(
        tok in joined
        for tok in ("holding", "color", "phone", "wearing", "blue")
    )


def test_suggested_questions_avoid_caption_duplicates() -> None:
    questions = generate_suggested_questions(_packet(), language="en", limit=3)
    joined = " ".join(q.lower() for q in questions)
    assert "what is the person sitting on" not in joined
