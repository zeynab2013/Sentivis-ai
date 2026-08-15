"""Final competition reliability — activity, venue, and caption↔QA consistency."""

from __future__ import annotations

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
from analysis.context.context_builder import ContextBuilder
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
from core.contracts.verified_evidence import ActivityEvidenceLevel
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.semantic.natural_caption_service import NaturalCaptionService
from language.validation.caption_factuality import (
    ClaimSupport,
    filter_unsupported_claims_verified,
)
from types import SimpleNamespace


def _ncs() -> NaturalCaptionService:
    return NaturalCaptionService(vision_model=SimpleNamespace())


def _ctx(
    nodes: tuple[SceneNode, ...],
    *,
    relations: tuple[Relation, ...] = (),
    activities: tuple[ActivityEvidence, ...] = (),
    setting: str = "indoor scene",
    scene_type: str = "indoor scene",
    indoor_outdoor: str = "indoor",
) -> SceneContext:
    attrs = tuple(Attribute(i, "confidence", "90%") for i in range(len(nodes)))
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=activities, confidence=0.7),
        environment=EnvironmentInfo(
            scene_type=scene_type,
            setting=setting,
            time_of_day="day",
            weather="unknown",
            indoor_outdoor=indoor_outdoor,
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes[:4]),
        spatial_summary="",
    )


def _answer(packet, question: str) -> str:
    session = VisionAssistantSession(image_key="final", evidence=packet)
    return VisionAssistant(client=None).answer(session, question, language="en")  # type: ignore[arg-type]


def test_racket_holding_is_not_playing_tennis() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "racket_1", "tennis racket", 0.05, "middle-right"),
        ),
        relations=(Relation(0, 1, "holding", 0.88),),
        activities=(
            ActivityEvidence(
                "playing tennis",
                0.90,
                (0, 1),
                ("holding",),
                "person holding racket",
            ),
        ),
        setting="outdoor scene",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        "playing tennis" in a.activity.lower()
        and a.evidence_level
        in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        for a in verified.activities
    )
    understanding = language_understanding_from_verified(verified)
    activity = _ncs()._infer_rich_activity(
        understanding, ["person #1"], "", "holding a tennis racket"
    )
    assert "tennis" not in activity.lower() or activity.lower().startswith("holding")
    answer = _answer(build_evidence_packet(verified_evidence=verified), "What is the person doing?")
    assert "playing tennis" not in answer.lower()


def test_laptop_alone_is_not_office_work() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "laptop_1", "laptop", 0.08, "middle-center"),
        ),
        relations=(Relation(0, 1, "near", 0.80),),
        activities=(
            ActivityEvidence(
                "working at a computer",
                0.85,
                (0, 1),
                ("near",),
                "laptop present",
            ),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        "working" in a.activity.lower() and a.qa_safe for a in verified.activities
    )
    setting = (verified.scene.setting or "").lower()
    scene_type = (verified.scene.scene_type or "").lower()
    assert "office" not in setting
    assert "office" not in scene_type


def test_kitchen_objects_alone_not_cooking() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "sink_1", "sink", 0.1, "middle-right"),
            SceneNode(2, "bowl_1", "bowl", 0.03, "middle-center"),
        ),
        relations=(),
        setting="kitchen",
        scene_type="kitchen",
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        any(tok in a.activity.lower() for tok in ("cook", "prepar"))
        and a.qa_safe
        for a in verified.activities
    )


def test_holding_cup_is_not_drinking() -> None:
    analyzer = HeuristicActivityAnalyzer(load_analysis_config())
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "cup_1", "cup", 0.02, "middle-center"),
        ),
        relations=(Relation(0, 1, "holding", 0.85),),
    )
    hints = analyzer.analyze(graph)
    assert not any(a.activity.lower() == "drinking" for a in hints.activities)


def test_near_dog_is_not_walking() -> None:
    analyzer = HeuristicActivityAnalyzer(load_analysis_config())
    graph = SceneGraph(
        nodes=(
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "dog_1", "dog", 0.1, "middle-right"),
        ),
        relations=(Relation(0, 1, "near", 0.80),),
    )
    hints = analyzer.analyze(graph)
    assert not any("walking" in a.activity.lower() for a in hints.activities)
    assert not any("pet interaction" in a.activity.lower() for a in hints.activities)


def test_single_horse_is_not_farm() -> None:
    builder = ContextBuilder(load_analysis_config())
    # Minimal graph through analyze path is heavy; call setting helper via public API shape.
    labels = {"person", "horse"}
    scene_type, setting = builder._specific_outdoor_setting(labels, set())
    assert "farm" not in scene_type.lower()
    assert "farm" not in setting.lower()


def test_bus_alone_is_not_highway() -> None:
    ctx = _ctx(
        (SceneNode(0, "bus_1", "bus", 0.4, "middle-center"),),
        setting="highway",
        scene_type="highway",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    assert "highway" not in (verified.scene.scene_type or "").lower()
    assert "highway" not in (verified.scene.setting or "").lower()


def test_unsupported_long_sentence_dropped() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.90),),
        activities=(
            ActivityEvidence(
                "riding a bicycle",
                0.90,
                (0, 1),
                ("riding",),
                "riding",
            ),
        ),
        setting="outdoor area",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    text = (
        "A person is riding a bicycle. "
        "Meanwhile the teacher is happily discussing plans for shopping on the highway."
    )
    cleaned = filter_unsupported_claims_verified(text, verified)
    lower = cleaned.lower()
    assert "riding" in lower
    assert "teacher" not in lower
    assert "shopping" not in lower
    assert "highway" not in lower


def test_caption_activity_must_be_qa_answerable() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.2, "middle-center"),
            SceneNode(1, "bicycle_1", "bicycle", 0.15, "middle-center"),
        ),
        relations=(Relation(0, 1, "riding", 0.90),),
        activities=(
            ActivityEvidence(
                "riding a bicycle",
                0.90,
                (0, 1),
                ("riding",),
                "riding",
            ),
        ),
        setting="outdoor area",
        scene_type="outdoor scene",
        indoor_outdoor="outdoor",
    )
    verified = build_verified_scene_evidence(ctx)
    understanding = language_understanding_from_verified(verified)
    acts = [
        f.value.lower()
        for f in understanding.facts
        if f.predicate == "activity"
    ]
    packet = build_evidence_packet(verified_evidence=verified)
    answer = _answer(packet, "What is the person doing?").lower()
    for act in acts:
        # Every caption-facing CONFIRMED activity must be recoverable in QA.
        key = next((t for t in act.split() if len(t) > 4), act)
        assert key in answer or any(
            a.qa_safe and key in a.activity.lower() for a in verified.activities
        )


def test_holding_bicycle_not_upgraded_to_riding_in_caption() -> None:
    service = _ncs()
    from core.contracts.reasoning import EvidenceFact, SceneUnderstanding

    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "detector"),
            EvidenceFact("bicycle #1", "is", "bicycle", 0.9, "detector"),
            EvidenceFact("person #1", "holding", "bicycle #1", 0.88, "relationships"),
        ),
        ranked_subjects=("person #1", "bicycle #1"),
        environment_keys=("outdoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    activity = service._infer_rich_activity(
        understanding, ["person #1"], "", "holding a bicycle"
    )
    assert activity == "holding a bicycle"
