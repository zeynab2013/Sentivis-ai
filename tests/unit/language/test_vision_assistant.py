"""Vision Assistant unit tests — evidence Q&A without VLM re-run."""

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
from language.assistant.evidence_packet import build_evidence_packet, retrieve_relevant_evidence
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import AssistantTurn, VisionAssistant, VisionAssistantSession
from language.refinement.caption_sanity import sanitize_caption


def _street_context() -> SceneContext:
    person = SceneNode(0, "obj-0", "person", 0.25, "middle-center")
    car = SceneNode(1, "obj-1", "car", 0.30, "middle-left")
    tree = SceneNode(2, "obj-2", "tree", 0.15, "background")
    return SceneContext(
        graph=SceneGraph(
            nodes=(person, car, tree),
            relations=(Relation(0, 1, "near", 0.7),),
        ),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "clothing_color", "dark gray"),
                Attribute(0, "shirt_color", "dark gray"),
                Attribute(1, "dominant_color", "maroon"),
            )
        ),
        activities=ActivityHints(
            activities=(ActivityEvidence("crossing a street", 0.6, (0, 1), ("near",), "street"),),
            confidence=0.6,
        ),
        environment=EnvironmentInfo(
            scene_type="street",
            setting="city street",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="public",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=("trees lining street",),
        ),
        object_count=3,
        dominant_objects=("person", "car", "tree"),
        spatial_summary="Person near car on street.",
    )


def test_suggested_questions_avoid_caption_duplicates() -> None:
    caption = (
        "A person wearing a dark gray jacket is crossing a city street beside a maroon car, "
        "with trees lining the street."
    )
    packet = build_evidence_packet(
        _street_context(),
        canonical_caption_en=caption,
        evidence_brief="person dark gray; car maroon",
        ocr_snippets=(),
    )
    questions = generate_suggested_questions(packet, language="en", limit=1)
    assert 0 <= len(questions) <= 4
    joined = " ".join(q.lower() for q in questions)
    assert "what is the person wearing" not in joined
    assert "what color is the car" not in joined
    # Zero suggestions is acceptable when every candidate is redundant or weak.
    assert all("describe the image" not in q.lower() for q in questions)


def test_sanitize_crossing_street_grammar() -> None:
    text = (
        "A person in a dark gray jacket is crossing street on a city street. "
        "Around them lies a city street, with trees lining the edge of the view."
    )
    cleaned = sanitize_caption(text)
    assert "crossing a" in cleaned.lower()
    assert "crossing street on" not in cleaned.lower()
    assert "around them lies" not in cleaned.lower()


def test_vision_assistant_no_vlm_and_multiturn(monkeypatch) -> None:
    packet = build_evidence_packet(
        _street_context(),
        canonical_caption_en="A person in a dark gray jacket is crossing a city street near a maroon car.",
        evidence_brief="person jacket dark gray; car maroon",
    )
    session = VisionAssistantSession(image_key="img1", evidence=packet)

    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 220, purpose: str = "assistant"):
            from analysis.activity.ollama_client import OllamaResponse

            question_line = ""
            for line in user.splitlines():
                if line.startswith("QUESTION:"):
                    question_line = line.lower()
                    break
            if "close" in question_line or "near the person" in question_line:
                return OllamaResponse(text="Yes, the car is near the person.", model="gemma3:4b")
            return OllamaResponse(text="Answered from evidence.", model="gemma3:4b")

    assistant = VisionAssistant(client=_StubClient())  # type: ignore[arg-type]
    # Color is answered directly from evidence (no caption lookup).
    a1 = assistant.answer(session, "What color is the car?", language="en")
    assert "maroon" in a1.lower()
    assert session.assistant_vlm_calls == 0
    a2 = assistant.answer(session, "Is it close to the person?", language="en")
    assert session.assistant_vlm_calls == 0
    assert len(session.turns) == 4
    assert a2


def test_unsupported_question_does_not_hallucinate_age(monkeypatch) -> None:
    packet = build_evidence_packet(_street_context(), canonical_caption_en="A person stands near a car.")
    session = VisionAssistantSession(image_key="img2", evidence=packet)

    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 220, purpose: str = "assistant"):
            from analysis.activity.ollama_client import OllamaResponse

            return OllamaResponse(
                text="The image does not provide enough information to determine the person's exact age.",
                model="gemma3:4b",
            )

    answer = VisionAssistant(client=_StubClient()).answer(  # type: ignore[arg-type]
        session, "What is the person's exact age?", language="en"
    )
    assert "age" in answer.lower()
    assert session.assistant_vlm_calls == 0


def test_retrieve_relevant_evidence_prefers_color_attrs() -> None:
    packet = build_evidence_packet(
        _street_context(),
        canonical_caption_en="A person near a car.",
    )
    block = retrieve_relevant_evidence(packet, "What color is the car?")
    assert "maroon" in block.lower()
    assert "caption summary" not in block.lower()
    assert "a person near a car" not in block.lower()


def test_unlimited_user_questions_counter() -> None:
    packet = build_evidence_packet(_street_context(), canonical_caption_en="A person near a car.")
    session = VisionAssistantSession(image_key="img3", evidence=packet)

    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 220, purpose: str = "assistant"):
            from analysis.activity.ollama_client import OllamaResponse

            return OllamaResponse(text="Answered.", model="gemma3:4b")

    assistant = VisionAssistant(client=_StubClient())  # type: ignore[arg-type]
    for i in range(12):
        assistant.answer(session, f"Question number {i}?", language="en")
    assert session.assistant_llm_calls == 12
    assert session.assistant_vlm_calls == 0
