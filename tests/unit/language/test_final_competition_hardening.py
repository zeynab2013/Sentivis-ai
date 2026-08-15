"""Final competition hardening regressions for QA / UI presentation."""

from __future__ import annotations

from pathlib import Path

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
from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem, build_evidence_packet
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


def _farm_packet(*, leading: bool = True, khaki: bool = True) -> AssistantEvidencePacket:
    attrs = [
        Attribute(0, "confidence", "90%"),
        Attribute(0, "visibility", "high"),
        Attribute(1, "confidence", "94%"),
        Attribute(1, "visibility", "high"),
        Attribute(2, "confidence", "85%"),
        Attribute(2, "visibility", "high"),
        Attribute(3, "confidence", "80%"),
        Attribute(3, "visibility", "high"),
        Attribute(4, "confidence", "89%"),
        Attribute(4, "visibility", "high"),
    ]
    if khaki:
        attrs.extend(
            [
                Attribute(0, "clothing_color", "khaki"),
                Attribute(0, "shirt_color", "khaki"),
            ]
        )
    relations = ()
    if leading:
        relations = (Relation(0, 1, "leading", 0.81),)
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=(
                SceneNode(0, "person_1", "person", 0.13, "middle-center"),
                SceneNode(1, "horse_1", "horse", 0.40, "middle-center"),
                SceneNode(2, "person_2", "person", 0.05, "back-center"),
                SceneNode(3, "horse_2", "horse", 0.05, "back-center"),
                SceneNode(4, "fire_1", "fire", 0.08, "bottom-left"),
            ),
            relations=relations,
        ),
        attributes=AttributeSet(attributes=tuple(attrs)),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="grassy field",
            time_of_day="day",
            weather="clear",
            indoor_outdoor="outdoor",
            social_context="farm",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("fire (confidence: 89%)",),
        ),
        object_count=5,
        dominant_objects=("horse", "person", "fire"),
        spatial_summary="Person leading horse near fire.",
    )
    return build_evidence_packet(
        ctx,
        canonical_caption_en=(
            "On a grassy field, a person in a black sweatshirt holds a rope while "
            "leading a large brown horse, while another person and another horse "
            "stand farther back in the field."
        ),
        evidence_brief=(
            "Entities\nPerson 1\nHorse 1\nAttributes\nRelationships\n"
            "Overall scene understanding"
        ),
    )


def test_fire_smoke_question_direct_not_evidence_dump() -> None:
    packet = _farm_packet()
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "Is there fire or smoke visible in the scene?"
    )
    lower = answer.lower()
    assert answer
    assert "yes" in lower
    assert "fire" in lower
    assert "breakdown" not in lower
    assert "entities" not in lower
    assert "person_1" not in lower
    assert "evidence packet" not in lower


def test_clothing_color_does_not_invent_tshirt() -> None:
    packet = _farm_packet(khaki=True)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What color clothing is the person wearing?"
    )
    lower = answer.lower()
    assert "t-shirt" not in lower
    assert "khaki" not in lower
    # Caption-grounded black clothing / sweatshirt is preferred over weak khaki.
    assert "black" in lower
    assert "sweatshirt" in lower or "clothing" in lower


def test_people_count_answer() -> None:
    packet = _farm_packet()
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "How many people are visible?"
    )
    assert "two people" in answer.lower() or "2 people" in answer.lower()


def test_animals_visible_answer() -> None:
    packet = _farm_packet()
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What animals are visible?"
    )
    lower = answer.lower()
    assert "horse" in lower
    assert "person_1" not in lower


def test_holding_requires_verified_interaction() -> None:
    packet = _farm_packet(leading=False)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "Is the person holding the horse?"
    )
    lower = answer.lower()
    assert "yes" not in lower
    assert "can't reliably" in lower or "does not provide enough" in lower


def test_holding_vs_leading_nuance() -> None:
    packet = _farm_packet(leading=True)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "Is the person holding the horse?"
    )
    lower = answer.lower()
    assert "yes." not in lower
    assert "leading" in lower
    assert "holding" in lower
    assert "cannot be confirmed" in lower or "can't reliably confirm" in lower


def test_leading_yes_when_verified() -> None:
    packet = _farm_packet(leading=True)
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "Is the person leading the horse?"
    )
    lower = answer.lower()
    assert "yes" in lower
    assert "leading" in lower


def test_unsupported_detail_unknown() -> None:
    packet = _farm_packet()
    session = VisionAssistantSession(image_key="farm-unknown", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What is the person's exact age?", language="en"
    )
    lower = answer.lower()
    assert "age" in lower or "can't" in lower or "cannot" in lower
    assert "breakdown" not in lower


def test_llm_dump_scrubbed_never_shown() -> None:
    packet = _farm_packet()
    session = VisionAssistantSession(image_key="farm-dump", evidence=packet)

    class _DumpClient:
        def generate_text(self, **kwargs):  # noqa: ANN003
            class R:
                text = (
                    "Okay, here's a breakdown of the information presented. "
                    "Entities (Key Objects): Person 1, Horse 1. Attributes: ... "
                    "Relationships: ... Overall scene understanding: farm."
                )

            return R()

    # Force LLM path with an open question that has selected evidence but no direct answer.
    assistant = VisionAssistant(client=_DumpClient())  # type: ignore[arg-type]
    # Monkeypatch retrieve to skip direct answer while keeping selected items.
    original = assistant._retriever.retrieve

    def _forced(packet, question):  # noqa: ANN001
        result = original(packet, question)
        from language.assistant.visual_evidence_retriever import EvidenceRetrievalResult

        return EvidenceRetrievalResult(
            question=result.question,
            selected=result.selected or packet.reliable_items()[:3],
            prompt_block=result.prompt_block,
            direct_answer_en="",
            has_reliable_match=True,
        )

    assistant._retriever.retrieve = _forced  # type: ignore[method-assign]
    answer = assistant.answer(session, "Describe the scene mood in detail.", language="en")
    lower = answer.lower()
    assert "breakdown" not in lower
    assert "person_1" not in lower
    assert "entities" not in lower
    assert "can't reliably" in lower or "cannot" in lower


def test_prompt_excludes_evidence_brief_inventory() -> None:
    packet = _farm_packet()
    prompt = VisualEvidenceRetriever().retrieve(
        packet, "Is there fire or smoke visible in the scene?"
    ).prompt_block
    lower = prompt.lower()
    assert "evidence brief:" not in lower
    assert "overall scene understanding" not in lower
    assert "person_1" not in lower


def test_main_page_has_language_selector_no_settings_nav() -> None:
    main = Path("streamlit_app/main.py").read_text(encoding="utf-8")
    assert "_render_main_language_selector" in main
    assert "main_language_select" in main
    # Competition nav must not present Dashboard/Settings options.
    assert 'nav_labels = [t("streamlit.nav.analyze")]' in main
    assert "streamlit.nav.dashboard" not in main.split("_render_sidebar")[1].split("return nav_label")[0] or (
        "nav_labels.extend" not in main.split("def _render_sidebar")[1].split("def _render_main_language")[0]
    )
