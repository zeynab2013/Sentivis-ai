"""Unit tests for sentence evidence analyzer."""

from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    Attribute,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from language.validation.sentence_evidence import SentenceEvidenceAnalyzer


def _context() -> SceneContext:
    graph = SceneGraph(
        nodes=(SceneNode(0, "obj-1", "person", 0.12, "middle-center"),),
        relations=(),
    )
    env = EnvironmentInfo(
        scene_type="street",
        setting="street",
        time_of_day="daytime",
        weather="clear",
        indoor_outdoor="outdoor",
        social_context="none",
        crowd_level="empty",
        scene_complexity="low",
        evidence=("One person detected.",),
    )
    return SceneContext(
        graph=graph,
        attributes=AttributeSet(
            attributes=(Attribute(0, "shirt_color", "blue"), Attribute(0, "dominant_color", "blue"))
        ),
        activities=ActivityHints(
            activities=(
                ActivityEvidence(
                    activity="walking",
                    confidence=0.7,
                    supporting_node_indices=(0,),
                    supporting_relation_types=(),
                    rationale="Person in outdoor street scene.",
                ),
            ),
            confidence=0.7,
        ),
        environment=env,
        object_count=1,
        dominant_objects=("person",),
        spatial_summary="One person detected.",
    )


def test_sentence_evidence_keeps_supported_sentence() -> None:
    analyzer = SentenceEvidenceAnalyzer()
    caption = "A person is present. A submarine is visible."
    filtered = analyzer.filter_supported(caption, _context())
    assert "person" in filtered.lower()
    assert "submarine" not in filtered.lower()


def test_sentence_evidence_assigns_sources() -> None:
    analyzer = SentenceEvidenceAnalyzer()
    analyzed = analyzer.analyze("A person is present in the street.", _context())
    assert analyzed
    assert analyzed[0].confidence >= 0.45
    assert analyzed[0].sources
