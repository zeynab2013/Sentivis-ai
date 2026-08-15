"""PART 2-D — Competition polish regressions for suggestions, QA, and claims."""

from __future__ import annotations

from dataclasses import replace

from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem, build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
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


def _env() -> EnvironmentInfo:
    return EnvironmentInfo(
        scene_type="indoor",
        setting="room",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="indoor",
        social_context="",
        crowd_level="few",
        scene_complexity="medium",
        evidence=(),
    )


def _ctx(nodes, relations=(), attrs=()) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=_env(),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes),
        spatial_summary="",
    )


def _packet(items: tuple[EvidenceItem, ...], caption: str = "") -> AssistantEvidencePacket:
    return AssistantEvidencePacket(
        objects=tuple(i.subject for i in items if i.kind == "object"),
        attributes=(),
        relations=tuple(f"{i.subject} {i.predicate} {i.value}" for i in items if i.kind == "relation"),
        activities=(),
        environment=("indoor_outdoor=outdoor",),
        ocr=(),
        evidence_brief="test",
        canonical_caption_en=caption,
        items=items,
        from_verified=True,
    )


def test_suggested_near_requires_spatial_relation() -> None:
    """Co-presence of table alone must not ask proximity questions."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "table_1", "table", 0.12, "bottom-center"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = replace(
        build_evidence_packet(verified_evidence=verified),
        canonical_caption_en="A person is standing in a room.",
    )
    qs = generate_suggested_questions(packet, language="en")
    joined = " ".join(qs).lower()
    assert "near the person" not in joined
    assert "positioned near" not in joined
    assert "positioned next to" not in joined


def test_suggested_holding_requires_interaction() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "rope_1", "rope", 0.05, "middle-center"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "80%")),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = replace(
        build_evidence_packet(verified_evidence=verified),
        canonical_caption_en="A person stands outdoors.",
    )
    qs = generate_suggested_questions(packet, language="en")
    assert not any("holding" in q.lower() for q in qs)


def test_near_fire_question_not_answered_as_person_near() -> None:
    packet = _packet(
        (
            EvidenceItem(
                kind="object",
                subject="person",
                predicate="is",
                value="person",
                confidence=0.9,
                object_index=0,
            ),
            EvidenceItem(
                kind="object",
                subject="fire",
                predicate="is",
                value="fire",
                confidence=0.85,
                object_index=1,
            ),
            EvidenceItem(
                kind="relation",
                subject="person",
                predicate="near",
                value="chair",
                confidence=0.8,
                object_index=0,
                relation_kind="SPATIAL",
            ),
        ),
        caption="A person stands near a fire.",
    )
    answer = VisualEvidenceRetriever().try_direct_answer(packet, "What is near the fire?")
    assert answer
    assert "chair" not in answer.lower()
    assert "not provide enough verified spatial" in answer.lower() or "fire" in answer.lower()


def test_holding_answer_insufficient_without_relation() -> None:
    packet = _packet(
        (
            EvidenceItem(
                kind="object",
                subject="person",
                predicate="is",
                value="person",
                confidence=0.9,
                object_index=0,
            ),
            EvidenceItem(
                kind="object",
                subject="rope",
                predicate="is",
                value="rope",
                confidence=0.8,
                object_index=1,
            ),
        ),
        caption="A person stands outdoors.",
    )
    answer = VisualEvidenceRetriever().try_direct_answer(packet, "What is the person holding?")
    assert "not provide enough verified" in answer.lower()


def test_holding_answer_uses_verified_interaction() -> None:
    packet = _packet(
        (
            EvidenceItem(
                kind="relation",
                subject="person",
                predicate="holding",
                value="cup",
                confidence=0.82,
                object_index=0,
                relation_kind="INTERACTION",
            ),
        ),
        caption="A person is outdoors.",
    )
    answer = VisualEvidenceRetriever().try_direct_answer(packet, "What is the person holding?")
    assert "cup" in answer.lower()


def test_no_fake_accuracy_claims_in_en_ui_copy() -> None:
    from pathlib import Path
    import json

    en = json.loads((Path("translations") / "en.json").read_text(encoding="utf-8"))
    blob = " ".join(str(v).lower() for v in en.values())
    for banned in ("100% accurate", "human-level", "state-of-the-art", "perfect accuracy"):
        assert banned not in blob
    assert "internal quality signals" in en["streamlit.quality.internal_note"].lower()


def test_vehicle_question_avoids_distance_claim() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "car_1", "car", 0.2, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.8),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    verified = build_verified_scene_evidence(ctx)
    packet = replace(
        build_evidence_packet(verified_evidence=verified),
        canonical_caption_en="A person stands on a street.",
    )
    qs = generate_suggested_questions(
        packet,
        language="en",
    )
    joined = " ".join(qs).lower()
    assert "how close" not in joined
