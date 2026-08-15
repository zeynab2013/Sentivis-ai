"""Compact evidence packet for Vision Assistant (no VLM re-perception).

Canonical path: VerifiedSceneEvidence → AssistantEvidencePacket.
Raw SceneContext is a legacy fallback only when verified evidence is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.analysis import SceneContext
from core.contracts.verified_evidence import VerifiedSceneEvidence

_ATTR_CONF_MIN = 0.55
_WEAK_VISIBILITY = {"low", "partial", "occluded", "poor", "unclear", "blurry"}


@dataclass(frozen=True)
class EvidenceItem:
    """One grounded visual fact with confidence for gating."""

    kind: str  # object | attribute | relation | activity | environment | ocr
    subject: str
    predicate: str
    value: str
    confidence: float
    visibility: str = ""
    object_index: int = -1
    entity_id: str = ""
    related_entity_id: str = ""
    claim_status: str = ""
    relation_kind: str = ""  # SPATIAL | INTERACTION | OTHER
    evidence_level: str = ""  # CONFIRMED | SUPPORTED | UNKNOWN (activities)

    @property
    def reliable(self) -> bool:
        if self.confidence < _ATTR_CONF_MIN:
            return False
        if self.visibility.lower() in _WEAK_VISIBILITY:
            return False
        value = (self.value or "").strip().lower()
        if value in {"", "unknown", "unlikely", "none", "n/a"}:
            return False
        if self.claim_status == "UNCERTAIN":
            return False
        if self.kind == "activity" and (self.evidence_level or "").upper() == "UNKNOWN":
            return False
        return True

    def as_line(self) -> str:
        vis = f", visibility={self.visibility}" if self.visibility else ""
        eid = f", id={self.entity_id}" if self.entity_id else ""
        kind = f", kind={self.relation_kind}" if self.relation_kind else ""
        return (
            f"{self.kind}| {self.subject}.{self.predicate}={self.value} "
            f"(conf={self.confidence:.2f}{vis}{eid}{kind})"
        )


@dataclass(frozen=True)
class AssistantEvidencePacket:
    """Reusable grounded evidence for question answering."""

    objects: tuple[str, ...]
    attributes: tuple[str, ...]
    relations: tuple[str, ...]
    activities: tuple[str, ...]
    environment: tuple[str, ...]
    ocr: tuple[str, ...]
    evidence_brief: str
    canonical_caption_en: str
    items: tuple[EvidenceItem, ...] = ()
    # When set, packet was built from the canonical verified layer.
    from_verified: bool = False

    def reliable_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.items if item.reliable)

    def as_prompt_block(self, *, include_caption: bool = True) -> str:
        reliable = self.reliable_items()
        lines = [
            "VISUAL EVIDENCE (source of truth — use ONLY these facts):",
            "OBJECTS:",
            ", ".join(self.objects) if self.objects else "(none)",
            "RELIABLE ATTRIBUTES:",
            "\n".join(i.as_line() for i in reliable if i.kind == "attribute")
            or "(none meeting confidence gate)",
            "RELATIONS:",
            "\n".join(self.relations) if self.relations else "(none)",
            "ACTIVITIES:",
            ", ".join(self.activities) if self.activities else "(none)",
            "ENVIRONMENT:",
            "\n".join(self.environment) if self.environment else "(none)",
            "OCR:",
            " | ".join(self.ocr) if self.ocr else "(none)",
            "EVIDENCE BRIEF:",
            self.evidence_brief or "(none)",
        ]
        if include_caption:
            lines.extend(
                [
                    "CAPTION SUMMARY (optional context only — NOT the knowledge base):",
                    self.canonical_caption_en or "(none)",
                ]
            )
        return "\n".join(lines)


def build_evidence_packet(
    scene_context: SceneContext | None = None,
    *,
    canonical_caption_en: str = "",
    evidence_brief: str = "",
    ocr_snippets: tuple[str, ...] = (),
    verified_evidence: VerifiedSceneEvidence | None = None,
) -> AssistantEvidencePacket:
    """Build assistant evidence from VerifiedSceneEvidence when available.

    Prefer verified_evidence. SceneContext alone is legacy fallback and still
    applies relation gates — it must not invent interaction from raw proximity.
    """
    if verified_evidence is not None:
        return _packet_from_verified(
            verified_evidence,
            canonical_caption_en=canonical_caption_en,
            evidence_brief=evidence_brief,
            ocr_snippets=ocr_snippets,
        )
    if scene_context is None:
        raise ValueError("build_evidence_packet requires verified_evidence or scene_context")
    # Legacy path: build verified first so Caption/QA never see ungated graph edges.
    from analysis.evidence.verified_evidence_builder import build_verified_scene_evidence

    verified = build_verified_scene_evidence(
        scene_context,
        understanding=None,
        ocr_snippets=ocr_snippets,
    )
    return _packet_from_verified(
        verified,
        canonical_caption_en=canonical_caption_en,
        evidence_brief=evidence_brief or verified.as_evidence_brief(),
        ocr_snippets=ocr_snippets or verified.ocr_text,
    )


def _packet_from_verified(
    verified: VerifiedSceneEvidence,
    *,
    canonical_caption_en: str = "",
    evidence_brief: str = "",
    ocr_snippets: tuple[str, ...] = (),
) -> AssistantEvidencePacket:
    items: list[EvidenceItem] = []
    objects: list[str] = []
    for ent in verified.entities:
        objects.append(
            f"{ent.entity_id}:{ent.label} (zone={ent.position_zone}, "
            f"area={ent.area_ratio:.2f}, conf={ent.confidence:.2f}, "
            f"narrative_safe={ent.narrative_safe})"
        )
        items.append(
            EvidenceItem(
                kind="object",
                subject=ent.label,
                predicate="detected",
                value="yes",
                confidence=ent.confidence,
                object_index=ent.object_index,
                entity_id=ent.entity_id,
                claim_status="OBSERVED" if ent.narrative_safe else "UNCERTAIN",
            )
        )

    attribute_lines: list[str] = []
    for attr in verified.qa_attributes():
        attribute_lines.append(
            f"{attr.entity_id} {attr.name}={attr.value} "
            f"(conf={attr.confidence:.2f}, status={attr.status.value})"
        )
        items.append(
            EvidenceItem(
                kind="attribute",
                subject=attr.entity_id.split("_")[0] if attr.entity_id else "object",
                predicate=attr.name,
                value=attr.value,
                confidence=attr.confidence,
                visibility=attr.visibility,
                object_index=attr.object_index,
                entity_id=attr.entity_id,
                claim_status=attr.status.value,
            )
        )

    relation_lines: list[str] = []
    for rel in verified.qa_relations():
        relation_lines.append(
            f"{rel.subject_id} {rel.relation_type} {rel.object_id} "
            f"(kind={rel.kind.value}, conf={rel.confidence:.2f}, tier={rel.verification_tier})"
        )
        sub_label = verified.entity_by_id(rel.subject_id)
        obj_label = verified.entity_by_id(rel.object_id)
        items.append(
            EvidenceItem(
                kind="relation",
                subject=sub_label.label if sub_label else rel.subject_id,
                predicate=rel.relation_type,
                value=obj_label.label if obj_label else rel.object_id,
                confidence=rel.confidence,
                object_index=rel.subject_index,
                entity_id=rel.subject_id,
                related_entity_id=rel.object_id,
                claim_status=rel.status.value,
                relation_kind=rel.kind.value,
            )
        )

    activities: list[str] = []
    for act in verified.activities:
        level = getattr(act, "evidence_level", None)
        level_name = level.value if level is not None else ""
        # QA packet includes CONFIRMED + SUPPORTED answerable activities.
        if level_name == "UNKNOWN" and not act.qa_safe:
            continue
        if level_name not in {"CONFIRMED", "SUPPORTED"} and not act.qa_safe:
            continue
        if level_name == "UNKNOWN":
            continue
        activities.append(
            f"{act.activity} (conf={act.confidence:.2f}, level={level_name or 'CONFIRMED'})"
        )
        # Prefer a person/animal actor entity as the activity subject when available.
        actor_id = ""
        for eid in act.entity_ids:
            ent = verified.entity_by_id(eid)
            if ent is not None and ent.label.lower() in {
                "person",
                "man",
                "woman",
                "child",
                "people",
                "skier",
                "rider",
                "horse",
                "dog",
                "cat",
            }:
                actor_id = eid
                break
        if not actor_id and act.entity_ids:
            actor_id = act.entity_ids[0]
        items.append(
            EvidenceItem(
                kind="activity",
                subject=actor_id.split("_")[0] if actor_id else "scene",
                predicate="activity",
                value=act.activity,
                confidence=act.confidence,
                entity_id=actor_id,
                object_index=act.object_indices[0] if act.object_indices else -1,
                claim_status=act.status.value,
                evidence_level=level_name or ("CONFIRMED" if act.qa_safe else "UNKNOWN"),
            )
        )

    environment: list[str] = []
    scene = verified.scene
    for pred, val in (
        ("scene_type", scene.scene_type),
        ("setting", scene.setting),
        ("indoor_outdoor", scene.indoor_outdoor),
        ("time_of_day", scene.time_of_day),
        ("weather", scene.weather),
        ("crowd", scene.crowd_level),
    ):
        if val:
            environment.append(f"{pred}={val}")
            items.append(
                EvidenceItem(
                    kind="environment",
                    subject="scene",
                    predicate=pred,
                    value=val,
                    confidence=scene.confidence,
                    claim_status=scene.status.value,
                )
            )
    for part in scene.evidence[:8]:
        if part:
            environment.append(part)
            items.append(
                EvidenceItem(
                    kind="environment",
                    subject="scene",
                    predicate="evidence",
                    value=part,
                    confidence=scene.confidence,
                )
            )

    ocr = tuple(s.strip() for s in (ocr_snippets or verified.ocr_text) if s and s.strip())
    for snippet in ocr:
        items.append(
            EvidenceItem(
                kind="ocr",
                subject="text",
                predicate="reads",
                value=snippet,
                confidence=0.75,
                claim_status="OBSERVED",
            )
        )

    # Hazards from environment evidence lines.
    existing_labels = {item.subject.lower() for item in items if item.kind == "object"}
    for line in scene.evidence:
        lower = line.lower()
        for label in ("fire", "smoke", "flame"):
            if label not in lower or label in existing_labels:
                continue
            conf = 0.7
            if "confidence:" in lower:
                try:
                    pct = lower.split("confidence:", 1)[1].strip().rstrip(").%")
                    conf = float(pct) / 100.0
                except ValueError:
                    conf = 0.7
            items.append(
                EvidenceItem(
                    kind="object",
                    subject=label,
                    predicate="detected",
                    value="yes",
                    confidence=conf,
                    visibility="high",
                    entity_id=f"{label}_1",
                    claim_status="OBSERVED",
                )
            )
            existing_labels.add(label)

    brief = (evidence_brief or "").strip() or verified.as_evidence_brief()
    return AssistantEvidencePacket(
        objects=tuple(objects),
        attributes=tuple(attribute_lines),
        relations=tuple(relation_lines),
        activities=tuple(activities),
        environment=tuple(environment),
        ocr=ocr,
        evidence_brief=brief,
        canonical_caption_en=(canonical_caption_en or "").strip(),
        items=tuple(items),
        from_verified=True,
    )


def _parse_confidence(value: str) -> float:
    text = (value or "").strip().replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return 0.6
    if number > 1.0:
        return max(0.0, min(1.0, number / 100.0))
    return max(0.0, min(1.0, number))


def retrieve_relevant_evidence(packet: AssistantEvidencePacket, question: str) -> str:
    """Delegate to VisualEvidenceRetriever (caption is never the knowledge base)."""
    from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever

    return VisualEvidenceRetriever().retrieve(packet, question).prompt_block


def find_attribute(
    packet: AssistantEvidencePacket,
    *,
    predicate: str,
    subject_tokens: tuple[str, ...] = (),
    require_reliable: bool = True,
) -> EvidenceItem | None:
    """Return the best matching attribute item, optionally confidence-gated."""
    pred = predicate.lower()
    best: EvidenceItem | None = None
    for item in packet.items:
        if item.kind != "attribute":
            continue
        if item.predicate.lower() != pred:
            continue
        if require_reliable and not item.reliable:
            continue
        subject_blob = f"{item.subject} {item.entity_id}".lower()
        if subject_tokens and not any(tok in subject_blob for tok in subject_tokens):
            continue
        if best is None or item.confidence > best.confidence:
            best = item
    return best
