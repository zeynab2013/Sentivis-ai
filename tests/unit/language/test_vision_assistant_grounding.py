"""Vision Assistant grounding + suggested-question confidence tests."""

from __future__ import annotations

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
from language.assistant.evidence_packet import (
    build_evidence_packet,
    find_attribute,
    retrieve_relevant_evidence,
)
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.refinement.caption_sanity import sanitize_caption


def _ski_context(*, shoe_color: str | None = None, shoe_visibility: str = "low") -> SceneContext:
    person = SceneNode(0, "obj-0", "person", 0.28, "middle-center")
    skis = SceneNode(1, "obj-1", "skis", 0.12, "bottom-center")
    attrs = [
        Attribute(0, "confidence", "88%"),
        Attribute(0, "visibility", "high"),
        Attribute(0, "clothing_color", "red"),
        Attribute(0, "clothing_type", "jacket"),
        Attribute(0, "jacket", "likely"),
        Attribute(1, "confidence", "81%"),
        Attribute(1, "visibility", "high"),
        Attribute(1, "dominant_color", "black"),
    ]
    if shoe_color is not None:
        attrs = [a for a in attrs if not (a.object_index == 0 and a.name == "visibility")]
        attrs.extend(
            [
                Attribute(0, "visibility", shoe_visibility),
                Attribute(0, "shoes_color", shoe_color),
            ]
        )
    return SceneContext(
        graph=SceneGraph(
            nodes=(person, skis),
            relations=(Relation(0, 1, "using", 0.8),),
        ),
        attributes=AttributeSet(attributes=tuple(attrs)),
        activities=ActivityHints(
            activities=(ActivityEvidence("skiing", 0.85, (0, 1), ("using",), "ski slope"),),
            confidence=0.85,
        ),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="snowy mountain slope",
            time_of_day="day",
            weather="snowy",
            indoor_outdoor="outdoor",
            social_context="sport",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("snow-covered slope",),
        ),
        object_count=2,
        dominant_objects=("person", "skis"),
        spatial_summary="Person skiing with skis on snowy slope.",
    )


def test_assistant_answers_from_evidence_absent_from_caption() -> None:
    thin_caption = "A person wearing a red jacket is skiing on a snowy outdoor slope."
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en=thin_caption,
        evidence_brief="person red jacket; skis; skiing",
    )
    session = VisionAssistantSession(image_key="ski1", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What equipment is the skier using?", language="en"
    )
    assert "ski" in answer.lower()
    assert "caption" not in answer.lower()
    assert session.assistant_vlm_calls == 0


def test_suggested_questions_reject_weak_shoe_color() -> None:
    packet = build_evidence_packet(
        _ski_context(shoe_color="black", shoe_visibility="low"),
        canonical_caption_en="A person wearing a red jacket is skiing on a snowy slope.",
    )
    shoe = find_attribute(packet, predicate="shoes_color", require_reliable=True)
    assert shoe is None
    questions = generate_suggested_questions(packet, language="en", limit=1)
    assert len(questions) <= 4
    joined = " ".join(q.lower() for q in questions)
    assert "shoe" not in joined


def test_suggested_questions_prefer_equipment_over_caption_duplicates() -> None:
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en="A person wearing a red jacket is skiing on a snowy outdoor slope.",
    )
    questions = generate_suggested_questions(packet, language="en", limit=1)
    assert len(questions) <= 4
    joined = " ".join(q.lower() for q in questions)
    assert "what color is the jacket" not in joined
    assert "what is the person wearing" not in joined
    assert "equipment" in joined


def test_retrieve_evidence_caption_is_not_primary() -> None:
    packet = build_evidence_packet(
        _ski_context(),
        canonical_caption_en="A short summary only.",
    )
    block = retrieve_relevant_evidence(packet, "What equipment is the skier using?")
    assert "skis" in block.lower()
    assert "source of truth" in block.lower() or "visual evidence facts" in block.lower()
    # Caption must not be injected into the QA prompt (prevents caption restatement).
    assert "optional caption" not in block.lower()
    assert "a short summary only" not in block.lower()


def test_scrub_caption_dependent_refusal() -> None:
    packet = build_evidence_packet(_ski_context(), canonical_caption_en="A person is skiing.")
    session = VisionAssistantSession(image_key="ski2", evidence=packet)

    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 220, purpose: str = "assistant"):
            from analysis.activity.ollama_client import OllamaResponse

            return OllamaResponse(
                text="The caption does not mention the ski poles.",
                model="gemma3:4b",
            )

    answer = VisionAssistant(client=_StubClient()).answer(  # type: ignore[arg-type]
        session, "Describe the lighting quality of this photograph?", language="en"
    )
    assert "caption" not in answer.lower()


def test_sanitize_ski_caption_fragments() -> None:
    bad = (
        "Skis is also visible in the scene. "
        "Snowy weather moves through the moment outdoors. "
        "Up close, the action is skiing."
    )
    cleaned = sanitize_caption(bad)
    assert "skis is" not in cleaned.lower()
    assert "moves through the moment" not in cleaned.lower()
    assert "up close, the action" not in cleaned.lower()


def test_estimated_age_blocked_from_evidence_packet() -> None:
    person = SceneNode(0, "obj-0", "person", 0.25, "middle-center")
    ctx = SceneContext(
        graph=SceneGraph(nodes=(person,), relations=()),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "90%"),
                Attribute(0, "visibility", "high"),
                Attribute(0, "estimated_age", "20-30"),
                Attribute(0, "clothing_color", "red"),
            )
        ),
        activities=ActivityHints(activities=(), confidence=0.5),
        environment=EnvironmentInfo(
            scene_type="outdoor",
            setting="slope",
            time_of_day="day",
            weather="snowy",
            indoor_outdoor="outdoor",
            social_context="sport",
            crowd_level="sparse",
            scene_complexity="simple",
            evidence=(),
        ),
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="person",
    )
    packet = build_evidence_packet(ctx, canonical_caption_en="A person wears a red jacket.")
    assert all("estimated_age" not in line.lower() for line in packet.attributes)
    block = retrieve_relevant_evidence(packet, "What is the person's exact age?")
    assert "20-30" not in block
    lower = block.lower()
    assert (
        "cannot be determined" in lower
        or "never reliably" in lower
        or "refuse age" in lower
        or "exact age cannot" in lower
    )
