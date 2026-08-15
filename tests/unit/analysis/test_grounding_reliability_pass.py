"""Grounding reliability: environment, activity, and hallucination claims."""

from __future__ import annotations

from analysis.activity.heuristic_activity_analyzer import HeuristicActivityAnalyzer
from analysis.context.context_builder import ContextBuilder
from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence
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
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.validation.caption_factuality import (
    filter_unsupported_claims_verified,
)


def _nodes(*labels: str) -> tuple[SceneNode, ...]:
    return tuple(
        SceneNode(i, f"{label}_{i}", label, 0.1 + i * 0.01, "middle-center")
        for i, label in enumerate(labels)
    )


def _builder() -> ContextBuilder:
    return ContextBuilder(load_analysis_config())


def _analyzer() -> HeuristicActivityAnalyzer:
    return HeuristicActivityAnalyzer(load_analysis_config())


def test_person_laptop_book_chair_is_not_classroom_or_lab() -> None:
    graph = SceneGraph(
        nodes=_nodes("person", "laptop", "book", "chair"),
        relations=(),
    )
    hints = ActivityHints(activities=(), confidence=0.0)
    env = _builder()._infer_environment(graph, hints)
    setting = (env.setting or "").lower()
    scene = (env.scene_type or "").lower()
    for banned in ("classroom", "laboratory", "library", "office"):
        assert banned not in setting
        assert banned not in scene
    assert "indoor" in setting or "indoor" in scene or scene in {"indoor scene", "unknown"}


def test_person_laptop_is_not_automatically_office() -> None:
    graph = SceneGraph(nodes=_nodes("person", "laptop"), relations=())
    env = _builder()._infer_environment(graph, ActivityHints(activities=(), confidence=0.0))
    assert "office" not in (env.setting or "").lower()
    assert "office" not in (env.scene_type or "").lower()


def test_person_laptop_near_is_not_working_or_studying() -> None:
    graph = SceneGraph(
        nodes=_nodes("person", "laptop"),
        relations=(Relation(0, 1, "near", 0.80),),
    )
    hints = _analyzer().analyze(graph)
    acts = " ".join(a.activity.lower() for a in hints.activities)
    assert "working" not in acts
    assert "studying" not in acts
    assert "student" not in acts
    assert "classroom" not in acts


def test_person_using_laptop_may_be_using_not_working() -> None:
    graph = SceneGraph(
        nodes=_nodes("person", "laptop"),
        relations=(Relation(0, 1, "using", 0.85),),
    )
    hints = _analyzer().analyze(graph)
    assert hints.activities
    joined = " ".join(a.activity.lower() for a in hints.activities)
    assert "using" in joined
    assert "studying" not in joined
    assert "student" not in joined


def test_book_with_chair_is_not_studying_without_interaction() -> None:
    graph = SceneGraph(
        nodes=_nodes("person", "book", "chair"),
        relations=(),
    )
    hints = _analyzer().analyze(graph)
    assert not any("studying" in a.activity.lower() for a in hints.activities)


def test_unsupported_classroom_studying_caption_raises_hallucination_risk() -> None:
    ctx = SceneContext(
        graph=SceneGraph(nodes=_nodes("person", "laptop", "book"), relations=()),
        attributes=AttributeSet(attributes=(Attribute(0, "confidence", "90%"),)),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="indoor scene",
            setting="indoor room",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=3,
        dominant_objects=("person", "laptop", "book"),
        spatial_summary="",
    )
    caption = "The student is studying in a classroom."
    report = CaptionQualityEvaluator().evaluate(caption, ctx)
    assert report.hallucination_risk is not None
    assert report.hallucination_risk >= 0.35
    assert report.overall_quality < 0.75


def test_filter_strips_invented_classroom_claims() -> None:
    ctx = SceneContext(
        graph=SceneGraph(nodes=_nodes("person", "laptop", "book"), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="indoor scene",
            setting="indoor room",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=3,
        dominant_objects=("person", "laptop", "book"),
        spatial_summary="",
    )
    verified = build_verified_scene_evidence(ctx)
    text = "The student is studying in a classroom with a laptop."
    cleaned = filter_unsupported_claims_verified(text, verified).lower()
    assert "classroom" not in cleaned
    assert "student" not in cleaned


def test_grounded_two_people_laptop_table_does_not_invent_school() -> None:
    ctx = SceneContext(
        graph=SceneGraph(
            nodes=_nodes("person", "person", "laptop", "dining table"),
            relations=(
                Relation(0, 3, "sitting_on", 0.82),
                Relation(0, 2, "near", 0.70),
            ),
        ),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(
            activities=(
                ActivityEvidence("sitting", 0.8, (0, 3), ("sitting_on",), "seated"),
            ),
            confidence=0.8,
        ),
        environment=EnvironmentInfo(
            scene_type="indoor scene",
            setting="indoor room",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="general",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=4,
        dominant_objects=("person", "laptop", "dining table"),
        spatial_summary="",
    )
    verified = build_verified_scene_evidence(ctx)
    setting = (verified.scene.setting or "").lower()
    for banned in ("classroom", "laboratory", "school", "office", "student", "teacher"):
        assert banned not in setting
    caption = (
        "Two people are in the scene. One is seated at a table with a laptop "
        "while the other stands nearby."
    )
    cleaned = filter_unsupported_claims_verified(caption, verified).lower()
    for banned in ("classroom", "laboratory", "student", "teacher", "school", "meeting"):
        assert banned not in cleaned


def test_device_selector_runtime_label_is_honest() -> None:
    from services.models.device_selector import DeviceSelector
    from core.config.loader import load_app_config

    selector = DeviceSelector(load_app_config())
    preferred = selector.preferred_device("cuda")
    label = selector.runtime_device_label()
    assert preferred in {"cuda", "cpu"}
    if preferred == "cpu":
        assert label == "cpu"
    else:
        assert label.startswith("cuda")
