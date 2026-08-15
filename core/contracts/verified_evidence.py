"""Canonical verified scene evidence — single source of truth for Caption + QA.

Raw SceneContext may retain weak detector/geometry links for visualization.
Language systems MUST consume VerifiedSceneEvidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClaimStatus(str, Enum):
    """How strongly a claim is supported."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNCERTAIN = "UNCERTAIN"


class ActivityEvidenceLevel(str, Enum):
    """Three-tier activity confidence for QA / suggestions (not caption redesign).

    CONFIRMED — direct interaction/relation evidence.
    SUPPORTED — multiple corroborating visual signals, no single strong relation.
    UNKNOWN — weak / co-occurrence only; do not answer or suggest.
    """

    CONFIRMED = "CONFIRMED"
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"


class RelationKind(str, Enum):
    """Spatial layout vs interaction — never conflate the two."""

    SPATIAL = "SPATIAL"
    INTERACTION = "INTERACTION"
    OTHER = "OTHER"


@dataclass(frozen=True)
class VerifiedEntity:
    """Stable entity identity through the language pipeline."""

    entity_id: str  # e.g. person_1
    object_index: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # x_min,y_min,x_max,y_max
    position_zone: str = ""
    area_ratio: float = 0.0
    narrative_safe: bool = True
    source: str = "detector"


@dataclass(frozen=True)
class VerifiedAttribute:
    """Attribute bound to a stable entity."""

    entity_id: str
    object_index: int
    name: str
    value: str
    confidence: float
    status: ClaimStatus
    source: str
    visibility: str = ""
    narrative_safe: bool = True
    qa_safe: bool = True


@dataclass(frozen=True)
class VerifiedRelation:
    """Relationship between two stable entities."""

    subject_id: str
    object_id: str
    subject_index: int
    object_index: int
    relation_type: str
    kind: RelationKind
    confidence: float
    status: ClaimStatus
    source: str
    narrative_safe: bool
    qa_safe: bool
    verification_tier: str = "LOW"  # HIGH | MEDIUM | LOW | UNCERTAIN


@dataclass(frozen=True)
class VerifiedActivity:
    """Activity involving one or more entities."""

    activity: str
    entity_ids: tuple[str, ...]
    object_indices: tuple[int, ...]
    confidence: float
    status: ClaimStatus
    source: str
    supporting_relations: tuple[str, ...] = ()
    narrative_safe: bool = True
    qa_safe: bool = True
    evidence_level: ActivityEvidenceLevel = ActivityEvidenceLevel.UNKNOWN

    @property
    def answerable(self) -> bool:
        """True when QA / suggestions may use this activity."""
        return self.evidence_level in {
            ActivityEvidenceLevel.CONFIRMED,
            ActivityEvidenceLevel.SUPPORTED,
        } or self.qa_safe


@dataclass(frozen=True)
class VerifiedSceneContext:
    """Environment / scene-level context with confidence."""

    indoor_outdoor: str = ""
    setting: str = ""
    scene_type: str = ""
    time_of_day: str = ""
    weather: str = ""
    crowd_level: str = ""
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    status: ClaimStatus = ClaimStatus.INFERRED


@dataclass(frozen=True)
class RejectedClaim:
    """Fact considered but rejected — for auditability."""

    subject: str
    predicate: str
    value: str
    confidence: float
    source: str
    reason: str


@dataclass(frozen=True)
class VerifiedSceneEvidence:
    """Authoritative evidence for caption generation and QA.

    Caption and QA must not reconstruct facts from raw SceneContext when this
    object is available.
    """

    entities: tuple[VerifiedEntity, ...]
    attributes: tuple[VerifiedAttribute, ...]
    relations: tuple[VerifiedRelation, ...]
    activities: tuple[VerifiedActivity, ...]
    scene: VerifiedSceneContext
    ocr_text: tuple[str, ...]
    evidence_brief: str
    overall_confidence: float
    rejected: tuple[RejectedClaim, ...] = ()
    ranked_entity_ids: tuple[str, ...] = ()

    @property
    def people_count(self) -> int:
        """Authoritative QA people count: narrative-safe person entities only."""
        person_labels = {"person", "people", "man", "woman", "child"}
        return sum(
            1
            for e in self.entities
            if e.narrative_safe and e.label.lower() in person_labels
        )

    def entity_by_id(self, entity_id: str) -> VerifiedEntity | None:
        for ent in self.entities:
            if ent.entity_id == entity_id:
                return ent
        return None

    def entity_by_index(self, index: int) -> VerifiedEntity | None:
        for ent in self.entities:
            if ent.object_index == index:
                return ent
        return None

    def narrative_relations(self) -> tuple[VerifiedRelation, ...]:
        return tuple(r for r in self.relations if r.narrative_safe)

    def qa_relations(self) -> tuple[VerifiedRelation, ...]:
        return tuple(r for r in self.relations if r.qa_safe)

    def interaction_relations(self) -> tuple[VerifiedRelation, ...]:
        return tuple(r for r in self.relations if r.kind == RelationKind.INTERACTION and r.qa_safe)

    def spatial_relations(self) -> tuple[VerifiedRelation, ...]:
        return tuple(r for r in self.relations if r.kind == RelationKind.SPATIAL and r.qa_safe)

    def narrative_attributes(self) -> tuple[VerifiedAttribute, ...]:
        return tuple(a for a in self.attributes if a.narrative_safe)

    def qa_attributes(self) -> tuple[VerifiedAttribute, ...]:
        return tuple(a for a in self.attributes if a.qa_safe)

    def compose_evidence_brief(self) -> str:
        """Always rebuild brief from verified fields (never from raw SceneContext)."""
        human = self.compose_human_scene_summary()
        lines: list[str] = []
        if human:
            lines.append(human)
        if self.entities:
            lines.append("Entities")
            for ent in self.entities:
                if not ent.narrative_safe:
                    continue
                lines.append(f"- {ent.entity_id}: {ent.label} (conf={ent.confidence:.2f})")
        attrs = self.narrative_attributes()
        if attrs:
            lines.append("Attributes")
            for attr in attrs[:12]:
                lines.append(
                    f"- {attr.entity_id}.{attr.name}={attr.value} "
                    f"[{attr.status.value}/{attr.source}]"
                )
        rels = self.narrative_relations()
        if rels:
            lines.append("Relationships")
            for rel in rels[:8]:
                lines.append(
                    f"- {rel.subject_id} {rel.relation_type} {rel.object_id} "
                    f"({rel.kind.value}, {rel.status.value}, conf={rel.confidence:.2f})"
                )
        confirmed = [
            a
            for a in self.activities
            if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.narrative_safe
        ]
        supported = [
            a
            for a in self.activities
            if a.evidence_level == ActivityEvidenceLevel.SUPPORTED and a.qa_safe
        ]
        if confirmed:
            lines.append("CONFIRMED Activities")
            for act in confirmed[:6]:
                who = ",".join(act.entity_ids) if act.entity_ids else "scene"
                lines.append(
                    f"- {who}: {act.activity} (conf={act.confidence:.2f})"
                )
        if supported:
            lines.append("SUPPORTED Activities")
            for act in supported[:4]:
                who = ",".join(act.entity_ids) if act.entity_ids else "scene"
                lines.append(
                    f"- {who}: {act.activity} (conf={act.confidence:.2f})"
                )
        if self.scene.indoor_outdoor or self.scene.setting:
            lines.append("Scene")
            if self.scene.indoor_outdoor:
                lines.append(f"- indoor_outdoor={self.scene.indoor_outdoor}")
            if self.scene.setting:
                lines.append(f"- setting={self.scene.setting}")
        if self.ocr_text:
            lines.append("OCR")
            lines.append("- " + " | ".join(self.ocr_text[:4]))
        return "\n".join(lines)

    def compose_human_scene_summary(self) -> str:
        """Structured human-readable summary — sole narrative source for language.

        Format:
          Objects: …
          Relations: …
          Environment: …
        """
        from collections import Counter

        narr_ents = [e for e in self.entities if e.narrative_safe]
        counts = Counter(e.label.lower() for e in narr_ents)
        object_bits: list[str] = []
        for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if count == 1:
                object_bits.append(label)
            else:
                object_bits.append(f"{count} {label}s" if not label.endswith("s") else f"{count} {label}")

        relation_bits: list[str] = []
        for rel in self.narrative_relations()[:10]:
            sub = self.entity_by_id(rel.subject_id)
            obj = self.entity_by_id(rel.object_id)
            sub_l = sub.label if sub is not None else rel.subject_id
            obj_l = obj.label if obj is not None else rel.object_id
            # Keep entity ids so caption/QA can bind roles.
            relation_bits.append(
                f"{rel.subject_id}({sub_l}) {rel.relation_type.replace('_', ' ')} "
                f"{rel.object_id}({obj_l})"
            )

        env_bits: list[str] = []
        if self.scene.indoor_outdoor:
            env_bits.append(self.scene.indoor_outdoor.replace("_", " "))
        if self.scene.setting and self.scene.setting.lower() not in {
            "unknown",
            "general",
            "indoor room",
            "outdoor area",
        }:
            env_bits.append(self.scene.setting.replace("_", " "))
        elif self.scene.indoor_outdoor and self.scene.indoor_outdoor.lower() == "outdoor":
            env_bits.append("outdoor")

        lines: list[str] = ["Verified Scene Summary"]
        lines.append(
            "Objects: " + (", ".join(object_bits) if object_bits else "none verified")
        )
        lines.append(
            "Relations: "
            + ("; ".join(relation_bits) if relation_bits else "none verified")
        )
        lines.append(
            "Environment: " + (", ".join(dict.fromkeys(env_bits)) if env_bits else "unspecified")
        )
        return "\n".join(lines)

    def as_evidence_brief(self) -> str:
        """Compact brief for prompts — verified content only."""
        stored = self.evidence_brief.strip()
        return stored if stored else self.compose_evidence_brief()
