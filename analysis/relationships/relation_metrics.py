"""Authoritative relationship evidence gating for metrics, caption, QA, and UI.

Spatial layout relations are distinct from interaction relations.
near ≠ holding ≠ talking_to.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.contracts.analysis import Relation, SceneGraph
from core.contracts.verified_evidence import RelationKind

# Interaction vocabulary — may enter narrative only when strongly verified.
_INTERACTION_RELATIONS = frozenset(
    {
        "holding",
        "sitting_on",
        "riding",
        "leading",
        "guiding",
        "carrying",
        "using",
        "playing_with",
        "eating",
        "wearing",
        "driving",
    }
)

# Speculative interactions — almost never narrative/QA safe from geometry alone.
_SPECULATIVE_INTERACTIONS = frozenset(
    {
        "talking_to",
        "looking_at",
        "standing_beside",
    }
)

# Spatial / layout — never promoted to interaction.
_SPATIAL_RELATIONS = frozenset(
    {
        "near",
        "next_to",
        "beside",
        "left_of",
        "right_of",
        "above",
        "below",
        "far",
        "overlapping",
        "near_vehicle",
        "behind",
        "in_front_of",
        "outside",
        "parked_beside",
        "inside",  # containment is spatial/layout (also structurally strong)
    }
)

# Containment is strong spatial — caption/QA may use at high conf.
_STRONG_SPATIAL = frozenset({"inside", "on", "above", "below", "behind", "in_front_of"})

# Backward-compatible names used by older callers / tests.
_CAPTION_SAFE_RELATIONS = _INTERACTION_RELATIONS | {"inside"}
_MEANINGFUL_RELATIONS = _INTERACTION_RELATIONS | _SPECULATIVE_INTERACTIONS | {"inside"}
_WEAK_RELATION_TYPES = _SPATIAL_RELATIONS - _STRONG_SPATIAL

_MIN_INTERACTION_CAPTION = 0.70
_MIN_INTERACTION_QA = 0.68
_MIN_METRICS = 0.62
_MIN_SPECULATIVE = 0.82
_MIN_SPATIAL_QA = 0.72
_MIN_SPATIAL_NARRATIVE = 0.78  # only strong spatial in captions


class RelationEvidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class GatedRelation:
    relation: Relation
    tier: RelationEvidenceTier
    status: str  # OBSERVED | INFERRED | UNCERTAIN
    kind: RelationKind = RelationKind.OTHER
    narrative_safe: bool = False
    qa_safe: bool = False


def relation_kind(relation_type: str) -> RelationKind:
    rel = (relation_type or "").lower().strip()
    if rel in _SPECULATIVE_INTERACTIONS or rel in _INTERACTION_RELATIONS:
        return RelationKind.INTERACTION
    if rel in _SPATIAL_RELATIONS:
        return RelationKind.SPATIAL
    return RelationKind.OTHER


def classify_relation(relation: Relation) -> GatedRelation:
    """Assign evidence tier, kind, and language-safety flags."""
    rel = (relation.relation_type or "").lower().strip()
    conf = float(relation.confidence)
    kind = relation_kind(rel)

    if rel in _SPECULATIVE_INTERACTIONS:
        if conf >= _MIN_SPECULATIVE:
            return GatedRelation(
                relation,
                RelationEvidenceTier.MEDIUM,
                "INFERRED",
                kind=RelationKind.INTERACTION,
                narrative_safe=False,
                qa_safe=False,
            )
        return GatedRelation(
            relation,
            RelationEvidenceTier.UNCERTAIN,
            "UNCERTAIN",
            kind=RelationKind.INTERACTION,
            narrative_safe=False,
            qa_safe=False,
        )

    if rel in _INTERACTION_RELATIONS:
        if conf >= _MIN_INTERACTION_CAPTION:
            return GatedRelation(
                relation,
                RelationEvidenceTier.HIGH,
                "OBSERVED",
                kind=RelationKind.INTERACTION,
                narrative_safe=True,
                qa_safe=True,
            )
        if conf >= _MIN_METRICS:
            return GatedRelation(
                relation,
                RelationEvidenceTier.MEDIUM,
                "INFERRED",
                kind=RelationKind.INTERACTION,
                narrative_safe=False,
                qa_safe=conf >= _MIN_INTERACTION_QA,
            )
        return GatedRelation(
            relation,
            RelationEvidenceTier.LOW,
            "UNCERTAIN",
            kind=RelationKind.INTERACTION,
            narrative_safe=False,
            qa_safe=False,
        )

    if rel in _SPATIAL_RELATIONS:
        # Spatial is never an interaction. Strong spatial can be narrative/QA.
        strong = rel in _STRONG_SPATIAL
        if strong and conf >= _MIN_SPATIAL_NARRATIVE:
            return GatedRelation(
                relation,
                RelationEvidenceTier.HIGH,
                "OBSERVED",
                kind=RelationKind.SPATIAL,
                narrative_safe=True,
                qa_safe=True,
            )
        if conf >= _MIN_SPATIAL_QA:
            return GatedRelation(
                relation,
                RelationEvidenceTier.MEDIUM,
                "INFERRED",
                kind=RelationKind.SPATIAL,
                narrative_safe=False,
                qa_safe=True,  # "is X near Y?" OK; never "holding"
            )
        return GatedRelation(
            relation,
            RelationEvidenceTier.LOW,
            "UNCERTAIN",
            kind=RelationKind.SPATIAL,
            narrative_safe=False,
            qa_safe=False,
        )

    return GatedRelation(
        relation,
        RelationEvidenceTier.UNCERTAIN,
        "UNCERTAIN",
        kind=kind,
        narrative_safe=False,
        qa_safe=False,
    )


def caption_safe_relations(graph: SceneGraph) -> tuple[Relation, ...]:
    """Relations allowed as definite narrative claims (interaction + strong spatial)."""
    return tuple(g.relation for g in (classify_relation(r) for r in graph.relations) if g.narrative_safe)


def qa_safe_relations(graph: SceneGraph) -> tuple[Relation, ...]:
    """Relations allowed in QA (includes verified spatial, excludes speculative talk/gaze)."""
    return tuple(g.relation for g in (classify_relation(r) for r in graph.relations) if g.qa_safe)


def meaningful_relations(graph: SceneGraph) -> tuple[Relation, ...]:
    """Metrics/UI: interactions at metrics floor + strong spatial; no weak near-as-action."""
    kept: list[Relation] = []
    for relation in graph.relations:
        gated = classify_relation(relation)
        if gated.kind == RelationKind.INTERACTION and gated.tier in {
            RelationEvidenceTier.HIGH,
            RelationEvidenceTier.MEDIUM,
        }:
            if relation.relation_type.lower() in _SPECULATIVE_INTERACTIONS:
                continue
            if relation.confidence >= _MIN_METRICS:
                kept.append(relation)
        elif gated.kind == RelationKind.SPATIAL and gated.qa_safe:
            kept.append(relation)
    return tuple(kept)


def count_meaningful_relations(graph: SceneGraph) -> int:
    return len(meaningful_relations(graph))


def is_weak_spatial(relation_type: str) -> bool:
    rel = (relation_type or "").lower()
    return rel in _SPATIAL_RELATIONS and rel not in _STRONG_SPATIAL


def is_speculative_interaction(relation_type: str) -> bool:
    return (relation_type or "").lower() in _SPECULATIVE_INTERACTIONS


def is_interaction(relation_type: str) -> bool:
    return relation_kind(relation_type) == RelationKind.INTERACTION


def is_spatial(relation_type: str) -> bool:
    return relation_kind(relation_type) == RelationKind.SPATIAL
