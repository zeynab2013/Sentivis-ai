"""Build VerifiedSceneEvidence from SceneReasoner + SceneContext.

This is the only approved path from raw analysis into Caption/QA language systems.
"""

from __future__ import annotations

import re
from collections import Counter

from analysis.relationships.relation_metrics import classify_relation
from core.contracts.analysis import SceneContext
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from core.contracts.verified_evidence import (
    ActivityEvidenceLevel,
    ClaimStatus,
    RejectedClaim,
    RelationKind,
    VerifiedActivity,
    VerifiedAttribute,
    VerifiedEntity,
    VerifiedRelation,
    VerifiedSceneContext,
    VerifiedSceneEvidence,
)
from vision.detection.narrative_gate import narrative_min_confidence

_PERSON = {"person", "people", "man", "woman", "child"}
_BLOCKED_ATTRS = {"estimated_age", "estimated_gender", "crop_description"}
_COLOR_ATTRS = {
    "clothing_color",
    "shirt_color",
    "shoes_color",
    "hair_color",
    "dominant_color",
    "secondary_color",
    "color",
    "pants_color",
}
# Clothing/person-appearance attrs must never bind to bowls/bikes/etc.
_CLOTHING_ATTR_NAMES = frozenset(
    {
        "shirt_color",
        "pants_color",
        "shoes_color",
        "clothing_color",
        "clothing_type",
        "clothing_style",
        "clothing_texture",
        "sleeve_length",
        "hair_color",
        "hair_length",
        "hairstyle",
        "footwear_type",
        "jacket",
        "coat",
        "dress",
        "hoodie",
        "blazer",
        "sweater",
        "skirt",
        "jeans",
        "shorts",
        "backpack",
        "handbag",
        "glasses",
        "sunglasses",
    }
)
_PERSON_LABELS_ATTR = frozenset({"person", "man", "woman", "child", "people", "skier", "rider"})


def _parse_confidence(raw: str | float | int | None) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower().replace("%", "")
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value / 100.0 if value > 1.0 else value


def _stable_entity_id(label: str, occurrence: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (label or "object").lower()).strip("_") or "object"
    return f"{slug}_{occurrence}"


def build_verified_scene_evidence(
    scene_context: SceneContext,
    understanding: SceneUnderstanding | None = None,
    *,
    ocr_snippets: tuple[str, ...] = (),
    vlm_caption: str | None = None,
    vlm_confidence: float = 0.75,
) -> VerifiedSceneEvidence:
    """Fuse SceneContext geometry with SceneUnderstanding facts into one verified object."""
    # Controlled VLM bridge: propose activities only when graph evidence corroborates.
    if (vlm_caption or "").strip():
        from analysis.evidence.vlm_activity_bridge import merge_vlm_activities_into_context

        scene_context = merge_vlm_activities_into_context(
            scene_context,
            vlm_caption or "",
            vlm_confidence=vlm_confidence,
        )
    graph = scene_context.graph
    rejected: list[RejectedClaim] = []

    # --- Detection confidences / visibility from attributes ---
    conf_by_index: dict[int, float] = {}
    vis_by_index: dict[int, str] = {}
    for item in scene_context.attributes.attributes:
        if item.name == "confidence":
            conf_by_index[item.object_index] = _parse_confidence(item.value)
        elif item.name == "visibility":
            vis_by_index[item.object_index] = (item.value or "").strip().lower()

    # --- Entities with stable IDs (person_1, horse_2, ...) ---
    label_counts: Counter[str] = Counter()
    entities: list[VerifiedEntity] = []
    index_to_id: dict[int, str] = {}
    for node in graph.nodes:
        label = (node.label or "object").lower().strip()
        label_counts[label] += 1
        entity_id = _stable_entity_id(label, label_counts[label])
        # Prefer SceneNode.object_id when already person_N style.
        if node.object_id and re.match(r"^[a-z]+_\d+$", node.object_id.lower()):
            entity_id = node.object_id.lower()
        conf = conf_by_index.get(node.index, 0.6)
        min_narr = narrative_min_confidence(label)
        narrative_safe = conf >= min_narr
        if not narrative_safe:
            rejected.append(
                RejectedClaim(
                    subject=entity_id,
                    predicate="detected",
                    value=label,
                    confidence=conf,
                    source="detector",
                    reason=f"below_narrative_floor_{min_narr:.2f}",
                )
            )
        index_to_id[node.index] = entity_id
        entities.append(
            VerifiedEntity(
                entity_id=entity_id,
                object_index=node.index,
                label=label,
                confidence=conf,
                bbox=None,
                position_zone=node.position_zone,
                area_ratio=node.bounding_box_area_ratio,
                narrative_safe=narrative_safe,
                source="detector",
            )
        )

    # Overlay fused object facts from SceneUnderstanding (source traceability).
    understanding_conf: dict[str, float] = {}
    if understanding is not None:
        for fact in understanding.facts:
            if fact.predicate != "is":
                continue
            key = fact.subject.split("#")[0].strip().lower()
            understanding_conf[key] = max(understanding_conf.get(key, 0.0), fact.confidence)

    # --- Attributes ---
    attributes: list[VerifiedAttribute] = []
    for item in scene_context.attributes.attributes:
        if item.name in {"confidence", "visibility"} or item.name in _BLOCKED_ATTRS:
            continue
        value = (item.value or "").strip()
        if value.lower() in {"", "unknown", "unlikely", "none", "n/a"}:
            continue
        entity = next((e for e in entities if e.object_index == item.object_index), None)
        if entity is None:
            continue
        # Never bind clothing/hair attributes onto non-person entities (bowl/bike bleed).
        if item.name in _CLOTHING_ATTR_NAMES and entity.label.lower() not in _PERSON_LABELS_ATTR:
            rejected.append(
                RejectedClaim(
                    subject=entity.entity_id,
                    predicate=item.name,
                    value=value,
                    confidence=conf_by_index.get(item.object_index, 0.55),
                    source="attributes",
                    reason="clothing_attr_on_non_person",
                )
            )
            continue
        conf = conf_by_index.get(item.object_index, 0.55)
        vis = vis_by_index.get(item.object_index, "")
        # Color / clothing inherit detection conf; shoes stricter.
        if item.name == "shoes_color":
            conf = min(conf, 0.40 if vis in {"", "unknown", "medium", "low", "partial"} else conf)
        status = ClaimStatus.OBSERVED if conf >= 0.70 else (
            ClaimStatus.INFERRED if conf >= 0.55 else ClaimStatus.UNCERTAIN
        )
        qa_safe = status != ClaimStatus.UNCERTAIN and conf >= 0.55
        narrative_safe = entity.narrative_safe and status in {
            ClaimStatus.OBSERVED,
            ClaimStatus.INFERRED,
        } and conf >= (0.70 if item.name in _COLOR_ATTRS else 0.58)
        # Aspect-ratio pose guesses are too weak for caption claims (kitchen "lying").
        if item.name == "pose":
            pose_val = value.lower()
            if pose_val in {"lying", "kneeling"}:
                narrative_safe = False
                status = ClaimStatus.UNCERTAIN
                qa_safe = False
            elif pose_val in {"standing", "sitting"}:
                # Standing/sitting from box ratio only — keep QA-optional, not caption fact.
                narrative_safe = False
                if conf < 0.80:
                    status = ClaimStatus.INFERRED
                    qa_safe = conf >= 0.62
        if status == ClaimStatus.UNCERTAIN or not entity.narrative_safe:
            rejected.append(
                RejectedClaim(
                    subject=entity.entity_id,
                    predicate=item.name,
                    value=value,
                    confidence=conf,
                    source="attributes",
                    reason="uncertain_or_entity_not_narrative_safe",
                )
            )
            if not qa_safe:
                continue
        # Pixel-crop attributes are the strongest color source (OBSERVED when conf high).
        color_source = "pixel_crop" if item.name in _COLOR_ATTRS or item.name.endswith("_color") else "attributes"
        attributes.append(
            VerifiedAttribute(
                entity_id=entity.entity_id,
                object_index=item.object_index,
                name=item.name,
                value=value,
                confidence=conf,
                status=status,
                source=color_source,
                visibility=vis,
                narrative_safe=narrative_safe,
                qa_safe=qa_safe,
            )
        )

    # Boost attribute status from SceneUnderstanding clothing/color facts.
    if understanding is not None:
        for fact in understanding.facts:
            if fact.predicate not in _COLOR_ATTRS and not fact.predicate.endswith("_color"):
                if fact.predicate not in {"clothing_type", "footwear_type"}:
                    continue
            subj = fact.subject
            # Match person #1 style to entity.
            match = re.search(r"#\s*(\d+)", subj)
            idx = int(match.group(1)) - 1 if match else -1
            entity = next((e for e in entities if e.object_index == idx), None)
            if entity is None:
                # Fall back to first matching label.
                label = subj.split("#")[0].strip().lower()
                entity = next((e for e in entities if e.label == label), None)
            if entity is None:
                continue
            # Never attach clothing attributes to non-person entities.
            if fact.predicate in _CLOTHING_ATTR_NAMES and entity.label.lower() not in _PERSON_LABELS_ATTR:
                rejected.append(
                    RejectedClaim(
                        subject=entity.entity_id,
                        predicate=fact.predicate,
                        value=str(fact.value),
                        confidence=fact.confidence,
                        source=fact.source,
                        reason="clothing_attr_on_non_person_from_reasoner",
                    )
                )
                continue
            # Reasoner/scene-context colors are never OBSERVED — pixel crops outrank them.
            status = (
                ClaimStatus.INFERRED
                if fact.confidence >= 0.60
                else ClaimStatus.UNCERTAIN
            )
            if status == ClaimStatus.UNCERTAIN:
                rejected.append(
                    RejectedClaim(
                        subject=entity.entity_id,
                        predicate=fact.predicate,
                        value=fact.value,
                        confidence=fact.confidence,
                        source=fact.source,
                        reason="understanding_fact_uncertain",
                    )
                )
                continue
            # Never overwrite a stronger OBSERVED crop color with a weaker reasoner guess.
            existing = next(
                (
                    a
                    for a in attributes
                    if a.entity_id == entity.entity_id and a.name == fact.predicate
                ),
                None,
            )
            if existing is not None:
                if existing.status == ClaimStatus.OBSERVED:
                    continue
                if existing.source in {"pixel_crop", "attributes"} and existing.confidence >= fact.confidence:
                    continue
            # Global/clothing aggregate colors must not replace entity-bound OBSERVED shirt/pants.
            if fact.predicate in {"clothing_color", "color", "dominant_color"}:
                observed_clothing = [
                    a
                    for a in attributes
                    if a.entity_id == entity.entity_id
                    and a.name in {"shirt_color", "clothing_color", "pants_color", "jacket_color"}
                    and a.status == ClaimStatus.OBSERVED
                ]
                if observed_clothing and str(fact.value).lower() != observed_clothing[0].value.lower():
                    rejected.append(
                        RejectedClaim(
                            subject=entity.entity_id,
                            predicate=fact.predicate,
                            value=str(fact.value),
                            confidence=fact.confidence,
                            source=fact.source or "scene_reasoner",
                            reason="weaker_inferred_color_blocked_by_observed_clothing",
                        )
                    )
                    continue
            # Replace/append verified attribute from reasoner.
            attributes = [
                a
                for a in attributes
                if not (a.entity_id == entity.entity_id and a.name == fact.predicate)
            ]
            attributes.append(
                VerifiedAttribute(
                    entity_id=entity.entity_id,
                    object_index=entity.object_index,
                    name=fact.predicate,
                    value=str(fact.value),
                    confidence=fact.confidence,
                    status=status,
                    source=fact.source or "scene_reasoner",
                    visibility=vis_by_index.get(entity.object_index, ""),
                    narrative_safe=entity.narrative_safe and status != ClaimStatus.UNCERTAIN,
                    qa_safe=True,
                )
            )

    # --- Relations: classify every graph edge; keep only verified ---
    relations: list[VerifiedRelation] = []
    for rel in graph.relations:
        gated = classify_relation(rel)
        sub_id = index_to_id.get(rel.subject_index, f"obj_{rel.subject_index}")
        obj_id = index_to_id.get(rel.object_index, f"obj_{rel.object_index}")
        if not gated.narrative_safe and not gated.qa_safe:
            rejected.append(
                RejectedClaim(
                    subject=sub_id,
                    predicate=rel.relation_type,
                    value=obj_id,
                    confidence=rel.confidence,
                    source="relationships",
                    reason=f"tier={gated.tier.value};kind={gated.kind.value}",
                )
            )
            continue
        # Never allow interaction claim if either entity is not narrative-safe for caption.
        narr = gated.narrative_safe
        if gated.kind == RelationKind.INTERACTION:
            sub_ent = next((e for e in entities if e.object_index == rel.subject_index), None)
            obj_ent = next((e for e in entities if e.object_index == rel.object_index), None)
            if sub_ent is not None and not sub_ent.narrative_safe:
                narr = False
            if obj_ent is not None and not obj_ent.narrative_safe:
                narr = False
        relations.append(
            VerifiedRelation(
                subject_id=sub_id,
                object_id=obj_id,
                subject_index=rel.subject_index,
                object_index=rel.object_index,
                relation_type=rel.relation_type.lower(),
                kind=gated.kind,
                confidence=rel.confidence,
                status=ClaimStatus(gated.status) if gated.status in ClaimStatus.__members__ else ClaimStatus.UNCERTAIN,
                source="relationships",
                narrative_safe=narr,
                qa_safe=gated.qa_safe,
                verification_tier=gated.tier.value,
            )
        )

    # --- Activities ---
    # Only promote activities with strong interaction support to QA-safe verified facts.
    # Co-occurrence heuristics (chair+tv → "office work", sink+person → "washing dishes")
    # remain possible signals but must not become authoritative CONFIRMED answers.
    activities: list[VerifiedActivity] = []
    _WEAK_SUPPORT = frozenset(
        {
            "near",
            "next_to",
            "beside",
            "standing_beside",
            "overlapping",
            "above",
            "below",
            "behind",
            "in_front_of",
            "left_of",
            "right_of",
            "inside",
            "on",
        }
    )
    _STRONG_SUPPORT = frozenset(
        {
            "holding",
            "using",
            "riding",
            "sitting_on",
            "looking_at",
            "playing",
            "playing_with",
            "carrying",
            "leading",
            "guiding",
            "eating",
            "drinking",
            "wearing",
            "pushing",
            "talking_to",
            "touching",
            "interacting",
            "reading",
            # VLM named-sport with independent multi-signal corroboration
            # (ball + people + field) — not proximity alone.
            "vlm_multisignal",
        }
    )
    # Possession / attention alone does not prove a performance activity.
    _POSSESSION_ONLY_SUPPORT = frozenset(
        {
            "holding",
            "playing_with",
            "wearing",
            "looking_at",
            "near",
            "next_to",
            "beside",
            "left_of",
            "right_of",
            "above",
            "below",
            "behind",
            "in_front_of",
            "overlapping",
        }
    )
    _ACTION_GRADE_SUPPORT = frozenset(
        {
            "using",
            "riding",
            "sitting_on",
            "playing",
            "carrying",
            "leading",
            "guiding",
            "eating",
            "drinking",
            "pushing",
            "talking_to",
            "touching",
            "interacting",
            "reading",
            "vlm_multisignal",
        }
    )
    # Furniture / display alone must not create SUPPORTED performance claims.
    _FURNITURE_DISPLAY = frozenset(
        {
            "chair",
            "couch",
            "sofa",
            "bed",
            "tv",
            "television",
            "dining table",
            "table",
            "bench",
            "remote",
        }
    )
    _PERSON_LABELS_LOCAL = frozenset(
        {"person", "man", "woman", "child", "people", "skier", "rider"}
    )

    def _is_performance_claim(name: str) -> bool:
        """True when the activity verb asserts performing an action, not mere presence."""
        a = (name or "").lower().strip()
        if not a:
            return False
        performance_markers = (
            "playing",
            "competing",
            "cooking",
            "preparing",
            "riding",
            "driving",
            "skiing",
            "snowboarding",
            "skating",
            "running",
            "cycling",
            "washing",
            "teaching",
            "working",
            "typing",
            "studying",
            "meeting",
            "office",
            "exercising",
            "dancing",
            "surfing",
            "skateboarding",
            "flying",
            "eating",
            "dining",
            "drinking",
            "feeding",
            "walking",
            "shopping",
            "crossing",
            "interaction",
            "campfire",
        )
        return any(a.startswith(m) or f" {m} " in f" {a} " for m in performance_markers)

    def _is_literal_interaction_activity(name: str) -> bool:
        """True when the activity is the interaction itself (not a venue/sport upgrade).

        'playing with a ball' / 'riding a bicycle' / 'holding a racket' are literal.
        Bare verbs like 'riding' without an object are NOT literal — they still need
        action-grade relation support (near ≠ riding).
        'playing tennis' / 'office work' / 'cooking' are not.
        """
        a = (name or "").lower().strip()
        if not a:
            return False
        prefixes = (
            "holding ",
            "playing with ",
            "carrying ",
            "using ",
            "riding ",
            "leading ",
            "guiding ",
            "pushing ",
            "sitting on ",
            "eating ",
            "drinking ",
        )
        return any(a.startswith(p) for p in prefixes)

    def _entity_label(eid: str) -> str:
        for e in entities:
            if e.entity_id == eid:
                return e.label.lower()
        return eid.split("_")[0]

    def _signal_count(
        *,
        has_person: bool,
        object_labels: set[str],
        support_types: tuple[str, ...],
        scene_type_hint: str,
        scene_setting_hint: str,
    ) -> int:
        """Generic multi-signal score (no category hard-codes)."""
        signals = 0
        if has_person:
            signals += 1
        task_objects = {lab for lab in object_labels if lab not in _FURNITURE_DISPLAY}
        if len(task_objects) >= 1:
            signals += 1
        if len(task_objects) >= 2:
            signals += 1
        if any(t in _ACTION_GRADE_SUPPORT for t in support_types):
            signals += 2
        elif any(t in _STRONG_SUPPORT for t in support_types):
            signals += 1
        elif any(t in _WEAK_SUPPORT for t in support_types):
            signals += 1
        st = (scene_type_hint or "").lower()
        se = (scene_setting_hint or "").lower()
        if st not in {"", "unknown", "general"} or se not in {"", "unknown", "general"}:
            if st not in {"indoor scene", "outdoor scene"} or se not in {
                "indoor room",
                "outdoor area",
            }:
                signals += 1
            elif st in {"indoor scene", "outdoor scene"}:
                signals += 0  # broad labels don't add signal
        return signals

    env_hint = scene_context.environment
    scene_type_hint = env_hint.scene_type or ""
    scene_setting_hint = env_hint.setting or ""
    labels_by_index = {node.index: (node.label or "").lower() for node in graph.nodes}

    # Deferred SUPPORTED candidates from multi-signal (not CONFIRMED).
    supported_pending: list[VerifiedActivity] = []

    for act in scene_context.activities.activities:
        if act.confidence < 0.55:
            rejected.append(
                RejectedClaim(
                    subject="scene",
                    predicate="activity",
                    value=act.activity,
                    confidence=act.confidence,
                    source="activity",
                    reason="below_activity_floor",
                )
            )
            continue
        support = tuple(t.lower() for t in act.supporting_relation_types)
        act_lower = (act.activity or "").lower().strip()
        # Intent / category overclaims: bags ≠ shopping; bicycle riding ≠ driving.
        if act_lower == "shopping":
            support_labels = {
                labels_by_index.get(i, "").lower()
                for i in act.supporting_node_indices
            }
            entity_labels = {
                e.label.lower()
                for e in entities
                if e.object_index in act.supporting_node_indices
            }
            if "shopping cart" not in support_labels and "shopping cart" not in entity_labels:
                rejected.append(
                    RejectedClaim(
                        subject="scene",
                        predicate="activity",
                        value=act.activity,
                        confidence=act.confidence,
                        source="activity",
                        reason="shopping_without_cart_evidence",
                    )
                )
                continue
        if act_lower == "driving":
            ride_labels = {
                e.label.lower()
                for e in entities
                if e.object_index in act.supporting_node_indices
            }
            ride_labels |= {
                labels_by_index.get(i, "").lower()
                for i in act.supporting_node_indices
            }
            if ride_labels & {"bicycle", "motorcycle", "scooter", "skateboard"} and not (
                ride_labels & {"car", "bus", "truck", "van", "train"}
            ):
                rejected.append(
                    RejectedClaim(
                        subject="scene",
                        predicate="activity",
                        value=act.activity,
                        confidence=act.confidence,
                        source="activity",
                        reason="driving_from_non_cabin_vehicle",
                    )
                )
                continue
            if "riding" in support and not (
                ride_labels & {"car", "bus", "truck", "van", "train"}
            ):
                rejected.append(
                    RejectedClaim(
                        subject="scene",
                        predicate="activity",
                        value=act.activity,
                        confidence=act.confidence,
                        source="activity",
                        reason="driving_from_riding_relation_alone",
                    )
                )
                continue
        strong_support = tuple(t for t in support if t in _STRONG_SUPPORT)
        action_grade = tuple(t for t in support if t in _ACTION_GRADE_SUPPORT)
        possession_only = bool(strong_support) and all(
            t in _POSSESSION_ONLY_SUPPORT for t in strong_support
        )
        weak_only = bool(support) and not strong_support and all(
            t in _WEAK_SUPPORT for t in support
        )
        empty_support = not support
        entity_ids = tuple(
            index_to_id[i] for i in act.supporting_node_indices if i in index_to_id
        )
        has_person_subject = any(
            (eid.split("_")[0] in _PERSON_LABELS_LOCAL)
            or any(
                e.entity_id == eid and e.label in _PERSON_LABELS_LOCAL for e in entities
            )
            for eid in entity_ids
        )
        object_labels = {
            e.label.lower()
            for e in entities
            if e.entity_id in entity_ids and e.label.lower() not in _PERSON_LABELS_LOCAL
        }
        has_object_evidence = bool(object_labels)
        soft_posture = act_lower in {
            "standing",
            "sitting",
            "walking",
            "present",
            "visible",
        }
        signals = _signal_count(
            has_person=has_person_subject,
            object_labels=object_labels,
            support_types=support,
            scene_type_hint=scene_type_hint,
            scene_setting_hint=scene_setting_hint,
        )

        if empty_support or weak_only:
            if soft_posture:
                if not has_person_subject or act.confidence < 0.70:
                    rejected.append(
                        RejectedClaim(
                            subject="scene",
                            predicate="activity",
                            value=act.activity,
                            confidence=act.confidence,
                            source="activity",
                            reason="posture_without_reliable_person_support",
                        )
                    )
                    continue
                activities.append(
                    VerifiedActivity(
                        activity=act.activity,
                        entity_ids=entity_ids,
                        object_indices=tuple(act.supporting_node_indices),
                        confidence=act.confidence,
                        status=ClaimStatus.INFERRED,
                        source="activity",
                        supporting_relations=support,
                        narrative_safe=True,
                        qa_safe=False,
                        evidence_level=ActivityEvidenceLevel.UNKNOWN,
                    )
                )
                continue

            # Multi-signal SUPPORTED path: never for furniture-only performance claims.
            task_objects = object_labels - _FURNITURE_DISPLAY
            furniture_only = bool(object_labels) and not task_objects
            if (
                signals >= 3
                and has_person_subject
                and task_objects
                and not furniture_only
                and act.confidence >= 0.60
            ):
                # Soften aggressive performance verbs that lack action-grade evidence.
                # Literal interaction phrases (playing with X, riding X) stay eligible.
                supported_name = act.activity
                if (
                    _is_performance_claim(act_lower)
                    and not action_grade
                    and not _is_literal_interaction_activity(act_lower)
                ):
                    # Weak/near co-occurrence must never promote upgraded performance claims.
                    rejected.append(
                        RejectedClaim(
                            subject="scene",
                            predicate="activity",
                            value=act.activity,
                            confidence=act.confidence,
                            source="activity",
                            reason="cooccurrence_without_strong_support",
                        )
                    )
                    continue
                supported_pending.append(
                    VerifiedActivity(
                        activity=supported_name,
                        entity_ids=entity_ids,
                        object_indices=tuple(act.supporting_node_indices),
                        confidence=min(act.confidence, 0.72),
                        status=ClaimStatus.INFERRED,
                        source="activity",
                        supporting_relations=support,
                        narrative_safe=False,  # do not alter caption facts
                        qa_safe=True,  # answerable for QA; caption still CONFIRMED-only
                        evidence_level=ActivityEvidenceLevel.SUPPORTED,
                    )
                )
                continue

            rejected.append(
                RejectedClaim(
                    subject="scene",
                    predicate="activity",
                    value=act.activity,
                    confidence=act.confidence,
                    source="activity",
                    reason="cooccurrence_without_strong_support",
                )
            )
            continue

        if not strong_support:
            rejected.append(
                RejectedClaim(
                    subject="scene",
                    predicate="activity",
                    value=act.activity,
                    confidence=act.confidence,
                    source="activity",
                    reason="unsupported_relation_types",
                )
            )
            continue

        if _is_performance_claim(act_lower) and possession_only and not action_grade:
            # Literal "playing with X" backed by playing_with is CONFIRMED interaction,
            # not a venue upgrade like "playing tennis".
            if _is_literal_interaction_activity(act_lower) and (
                "playing_with" in strong_support
                or "holding" in strong_support
                or "using" in strong_support
                or "carrying" in strong_support
            ):
                pass  # fall through to CONFIRMED / SUPPORTED path below
            else:
                # Holding equipment ≠ performing the sport — but the holding itself is CONFIRMED.
                held_labels = sorted(object_labels)
                if held_labels and "holding" in strong_support:
                    obj_name = held_labels[0]
                    article = "an" if obj_name[:1] in "aeiou" else "a"
                    activities.append(
                        VerifiedActivity(
                            activity=f"holding {article} {obj_name}",
                            entity_ids=entity_ids,
                            object_indices=tuple(act.supporting_node_indices),
                            confidence=min(0.88, max(act.confidence, 0.75)),
                            status=ClaimStatus.OBSERVED,
                            source="activity",
                            supporting_relations=support,
                            narrative_safe=True,
                            qa_safe=True,
                            evidence_level=ActivityEvidenceLevel.CONFIRMED,
                        )
                    )
                rejected.append(
                    RejectedClaim(
                        subject="scene",
                        predicate="activity",
                        value=act.activity,
                        confidence=act.confidence,
                        source="activity",
                        reason="performance_claim_from_possession_only",
                    )
                )
                continue

        # Strong interaction-backed activity → CONFIRMED when multi-signal enough.
        if not has_person_subject and act_lower not in {
            "transportation scene",
            "static scene",
        }:
            actor_ok = any(
                e.label
                in _PERSON_LABELS_LOCAL
                | {
                    "horse",
                    "dog",
                    "cat",
                    "cow",
                    "bird",
                    "sheep",
                    "bear",
                    "elephant",
                    "zebra",
                    "giraffe",
                }
                for e in entities
                if e.entity_id in entity_ids
            )
            if not actor_ok:
                rejected.append(
                    RejectedClaim(
                        subject="scene",
                        predicate="activity",
                        value=act.activity,
                        confidence=act.confidence,
                        source="activity",
                        reason="activity_missing_actor_entity",
                    )
                )
                continue

        status = ClaimStatus.OBSERVED if act.confidence >= 0.75 else ClaimStatus.INFERRED
        confirmed = act.confidence >= 0.68 and bool(strong_support)
        if confirmed and _is_performance_claim(act_lower):
            # Literal interaction activities need the matching relation, not a
            # stricter action-grade upgrade (playing_with backs "playing with a ball").
            if _is_literal_interaction_activity(act_lower):
                confirmed = has_object_evidence and bool(strong_support)
            else:
                confirmed = bool(action_grade) and has_object_evidence
        if confirmed:
            activities.append(
                VerifiedActivity(
                    activity=act.activity,
                    entity_ids=entity_ids,
                    object_indices=tuple(act.supporting_node_indices),
                    confidence=act.confidence,
                    status=status,
                    source="activity",
                    supporting_relations=support,
                    narrative_safe=True,
                    qa_safe=True,
                    evidence_level=ActivityEvidenceLevel.CONFIRMED,
                )
            )
        elif (
            act.confidence >= 0.62
            and has_person_subject
            and has_object_evidence
            and signals >= 3
            and not (
                _is_performance_claim(act_lower)
                and not action_grade
                and not _is_literal_interaction_activity(act_lower)
            )
        ):
            activities.append(
                VerifiedActivity(
                    activity=act.activity,
                    entity_ids=entity_ids,
                    object_indices=tuple(act.supporting_node_indices),
                    confidence=act.confidence,
                    status=ClaimStatus.INFERRED,
                    source="activity",
                    supporting_relations=support,
                    narrative_safe=False,
                    qa_safe=True,
                    evidence_level=ActivityEvidenceLevel.SUPPORTED,
                )
            )
        else:
            rejected.append(
                RejectedClaim(
                    subject="scene",
                    predicate="activity",
                    value=act.activity,
                    confidence=act.confidence,
                    source="activity",
                    reason="insufficient_multi_signal_support",
                )
            )

    # Synthesize CONFIRMED literal actions from verified INTERACTION relations
    # when heuristics did not already cover that person with a CONFIRMED activity.
    existing_act_text = {" ".join((a.activity or "").lower().split()) for a in activities}
    # Deduplicate / supersede weaker possession verbs when an action-grade
    # interaction already binds the same person–object pair (riding > holding).
    _SUPERSEDED_BY_ACTION = frozenset({"holding", "carrying", "wearing", "using"})
    _ACTION_SUPERSEDES = frozenset({"riding", "leading", "guiding", "pushing", "sitting_on"})
    action_pairs = {
        (rel.subject_id, rel.object_id)
        for rel in relations
        if rel.relation_type.lower().replace(" ", "_") in _ACTION_SUPERSEDES
        and (rel.qa_safe or rel.narrative_safe)
    }
    if action_pairs:
        filtered_rels: list[VerifiedRelation] = []
        for rel in relations:
            pred = rel.relation_type.lower().replace(" ", "_")
            if (
                pred in _SUPERSEDED_BY_ACTION
                and (rel.subject_id, rel.object_id) in action_pairs
            ):
                rejected.append(
                    RejectedClaim(
                        subject=rel.subject_id,
                        predicate=rel.relation_type,
                        value=rel.object_id,
                        confidence=rel.confidence,
                        source="relationships",
                        reason="superseded_by_action_grade_relation",
                    )
                )
                continue
            filtered_rels.append(rel)
        relations = filtered_rels

    persons_with_confirmed = {
        eid
        for a in activities
        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED and a.source == "activity"
        for eid in a.entity_ids
        if eid.startswith("person") or _entity_label(eid) in _PERSON_LABELS_LOCAL
    }
    for rel in relations:
        if not rel.qa_safe or rel.kind != RelationKind.INTERACTION:
            continue
        pred = (rel.relation_type or "").lower().replace(" ", "_")
        if pred not in _STRONG_SUPPORT:
            continue
        sub = next((e for e in entities if e.entity_id == rel.subject_id), None)
        obj = next((e for e in entities if e.entity_id == rel.object_id), None)
        if sub is None or obj is None:
            continue
        sub_lab = sub.label.lower()
        obj_lab = obj.label.lower()
        person_is_subject = sub_lab in _PERSON_LABELS_LOCAL
        if not person_is_subject:
            continue
        # Heuristic CONFIRMED already answers for this person — don't dilute with literals.
        if rel.subject_id in persons_with_confirmed:
            continue
        # Phrase the observed interaction literally.
        verb = pred.replace("_", " ")
        article = "an" if obj_lab[:1] in "aeiou" else "a"
        if pred in {"holding", "carrying", "using", "pushing"}:
            phrase = f"{verb} {article} {obj_lab}"
        elif pred in {"leading", "guiding", "riding"}:
            phrase = f"{verb} {article} {obj_lab}"
        elif pred == "looking_at":
            phrase = f"looking at {article} {obj_lab}"
        elif pred in {"talking_to", "interacting"}:
            phrase = f"{verb} {article} {obj_lab}"
        else:
            phrase = f"{verb} {article} {obj_lab}"
        key = " ".join(phrase.split())
        # Skip if a CONFIRMED activity already mentions this object+verb.
        if any(
            obj_lab in a.activity.lower() and verb.split()[0] in a.activity.lower()
            for a in activities
            if a.evidence_level == ActivityEvidenceLevel.CONFIRMED
        ):
            continue
        if key in existing_act_text:
            continue
        existing_act_text.add(key)
        # Riding / leading / holding are defining caption events — allow narrative use.
        # Holding equipment is literal INTERACTION evidence (not a sport-name upgrade).
        narr_ok = pred in {
            "riding",
            "leading",
            "guiding",
            "pushing",
            "holding",
            "carrying",
            "using",
        }
        activities.append(
            VerifiedActivity(
                activity=phrase,
                entity_ids=(rel.subject_id, rel.object_id),
                object_indices=(rel.subject_index, rel.object_index),
                confidence=max(rel.confidence, 0.78),
                status=ClaimStatus.OBSERVED,
                source="relation",
                supporting_relations=(pred,),
                narrative_safe=narr_ok,
                qa_safe=True,
                evidence_level=ActivityEvidenceLevel.CONFIRMED,
            )
        )

    # Append SUPPORTED candidates that do not collide with CONFIRMED phrases.
    confirmed_blobs = " ".join(
        a.activity.lower()
        for a in activities
        if a.evidence_level == ActivityEvidenceLevel.CONFIRMED
    )
    for pending in supported_pending:
        tokens = [
            t
            for t in re.findall(r"[a-z]{4,}", pending.activity.lower())
            if t not in {"with", "from", "scene", "person"}
        ]
        if tokens and any(t in confirmed_blobs for t in tokens):
            continue
        activities.append(pending)

    # --- Scene context ---
    env = scene_context.environment
    scene_conf = 0.7 if env.indoor_outdoor not in {"", "unknown"} else 0.4
    scene_type = env.scene_type or ""
    setting = env.setting or ""
    evidence_blob = _environment_evidence_blob(
        understanding,
        vlm_caption=vlm_caption,
        env_evidence=tuple(env.evidence or ()),
    )
    scene_type, setting, scene_conf = _calibrate_scene_label(
        scene_type,
        setting,
        scene_conf,
        entities=entities,
        activities=activities,
        evidence_blob=evidence_blob,
        indoor_outdoor=env.indoor_outdoor or "",
    )
    scene = VerifiedSceneContext(
        indoor_outdoor=env.indoor_outdoor or "",
        setting=setting,
        scene_type=scene_type,
        time_of_day="" if env.time_of_day in {"", "unknown"} else env.time_of_day,
        weather="" if env.weather in {"", "unknown"} else env.weather,
        crowd_level=env.crowd_level or "",
        confidence=scene_conf,
        evidence=tuple(env.evidence[:8]),
        status=ClaimStatus.OBSERVED if scene_conf >= 0.65 else ClaimStatus.INFERRED,
    )

    # Scene-grounded SUPPORTED actions when heuristics produced nothing answerable.
    # Driven by object clusters + person presence — not venue-name hard-codes.
    has_person_answerable = any(
        a.evidence_level
        in {ActivityEvidenceLevel.CONFIRMED, ActivityEvidenceLevel.SUPPORTED}
        and any(
            eid.startswith("person")
            or _entity_label(eid) in _PERSON_LABELS_LOCAL
            for eid in a.entity_ids
        )
        for a in activities
    )
    if not has_person_answerable:
        persons = [
            e
            for e in entities
            if e.narrative_safe and e.label.lower() in _PERSON_LABELS_LOCAL
        ]
        task = [
            e
            for e in entities
            if e.narrative_safe
            and e.label.lower() not in _PERSON_LABELS_LOCAL
            and e.label.lower() not in _FURNITURE_DISPLAY
        ]
        task_labels = {e.label.lower() for e in task}
        place = (setting or scene_type or "").strip()
        specific_place = place.lower() not in {
            "",
            "unknown",
            "general",
            "indoor scene",
            "outdoor scene",
            "indoor room",
            "outdoor area",
            "urban environment",
            "natural environment",
        }
        person_idxs = {p.object_index for p in persons}
        task_idxs = {t.object_index for t in task}
        linked = any(
            (
                r.subject_index in person_idxs and r.object_index in task_idxs
            )
            or (
                r.object_index in person_idxs and r.subject_index in task_idxs
            )
            for r in relations
            if r.qa_safe or r.narrative_safe
        )
        # Generic object-cluster → soft action phrases (not venue hard-codes).
        _CLUSTERS: tuple[tuple[frozenset[str], str], ...] = (
            (
                frozenset(
                    {
                        "oven",
                        "refrigerator",
                        "sink",
                        "microwave",
                        "toaster",
                        "bowl",
                        "cup",
                        "bottle",
                        "knife",
                        "fork",
                        "spoon",
                    }
                ),
                "preparing food",
            ),
            (
                frozenset({"laptop", "keyboard", "mouse", "monitor"}),
                "using a computer",
            ),
            (frozenset({"cell phone"}), "using a phone"),
        )
        cluster_phrase = ""
        for labels, phrase in _CLUSTERS:
            if len(task_labels & labels) >= 2 or (
                len(labels) == 1 and task_labels & labels and linked
            ):
                cluster_phrase = phrase
                break
        if (
            persons
            and len(task) >= 2
            and cluster_phrase
            and linked
            and (specific_place and scene_conf >= 0.65)
        ):
            actor = persons[0]
            if place and specific_place:
                activity_phrase = f"{cluster_phrase} in the {place}"
            else:
                activity_phrase = cluster_phrase
            activities.append(
                VerifiedActivity(
                    activity=activity_phrase,
                    entity_ids=(actor.entity_id, *(t.entity_id for t in task[:3])),
                    object_indices=(actor.object_index, *(t.object_index for t in task[:3])),
                    confidence=0.66,
                    status=ClaimStatus.INFERRED,
                    source="multi_signal",
                    supporting_relations=(),
                    narrative_safe=False,
                    qa_safe=True,
                    evidence_level=ActivityEvidenceLevel.SUPPORTED,
                )
            )

    ocr = tuple(s.strip() for s in (ocr_snippets or ()) if s and s.strip())
    if understanding is not None and understanding.ocr_text:
        ocr = tuple(dict.fromkeys((*ocr, *understanding.ocr_text)))

    brief = ""
    if understanding is not None and understanding.evidence_brief.strip():
        brief = understanding.evidence_brief.strip()

    overall = (
        float(understanding.overall_confidence)
        if understanding is not None
        else (
            sum(e.confidence for e in entities) / max(1, len(entities)) if entities else 0.0
        )
    )

    # Ranked entity IDs from understanding subjects when possible.
    ranked: list[str] = []
    if understanding is not None:
        for subject in understanding.ranked_subjects:
            if subject in {"scene", "vlm"}:
                continue
            match = re.search(r"#\s*(\d+)", subject)
            if match:
                idx = int(match.group(1)) - 1
                ent = next((e for e in entities if e.object_index == idx), None)
                if ent is not None:
                    ranked.append(ent.entity_id)
                    continue
            label = subject.split("#")[0].strip().lower()
            ent = next((e for e in entities if e.label == label and e.entity_id not in ranked), None)
            if ent is not None:
                ranked.append(ent.entity_id)
    if not ranked:
        ranked = [e.entity_id for e in entities if e.narrative_safe]

    draft = VerifiedSceneEvidence(
        entities=tuple(entities),
        attributes=tuple(attributes),
        relations=tuple(relations),
        activities=tuple(activities),
        scene=scene,
        ocr_text=ocr,
        evidence_brief="",
        overall_confidence=overall,
        rejected=tuple(rejected),
        ranked_entity_ids=tuple(ranked),
    )
    # Authoritative brief is always composed from verified fields.
    # Optional reasoner brief may append only when it does not introduce new relation verbs.
    composed = draft.compose_evidence_brief()
    if brief and not _brief_introduces_unverified_relations(brief, draft):
        composed = f"{composed}\nNotes\n- {brief[:400]}" if composed else brief[:500]
    return VerifiedSceneEvidence(
        entities=draft.entities,
        attributes=draft.attributes,
        relations=draft.relations,
        activities=draft.activities,
        scene=draft.scene,
        ocr_text=draft.ocr_text,
        evidence_brief=composed,
        overall_confidence=draft.overall_confidence,
        rejected=draft.rejected,
        ranked_entity_ids=draft.ranked_entity_ids,
    )


def _environment_evidence_blob(
    understanding: SceneUnderstanding | None,
    *,
    vlm_caption: str | None = None,
    env_evidence: tuple[str, ...] = (),
) -> str:
    """Collect scene-level text used to preserve specific environment labels."""
    parts: list[str] = []
    if vlm_caption:
        parts.append(vlm_caption)
    parts.extend(env_evidence)
    if understanding is not None:
        parts.append(understanding.evidence_brief or "")
        parts.extend(understanding.environment_keys or ())
        for fact in understanding.facts:
            if fact.subject in {"vlm", "scene"} or fact.predicate in {
                "observation",
                "setting",
                "scene_type",
                "indoor_outdoor",
            }:
                parts.append(f"{fact.predicate} {fact.value}")
    return " ".join(p for p in parts if p).lower()


def _calibrate_scene_label(
    scene_type: str,
    setting: str,
    confidence: float,
    *,
    entities: list[VerifiedEntity],
    activities: list[VerifiedActivity],
    evidence_blob: str = "",
    indoor_outdoor: str = "",
) -> tuple[str, str, float]:
    """Downgrade over-specific venue labels when entity evidence is weak/conflicting.

    General rule: do not assert restaurant/office/classroom/highway/court/etc. from
    a single object class. Prefer broader indoor/outdoor/natural/urban labels when
    distinctive venue cues are absent.

    Also: do not destroy useful verified scene specificity (e.g. trail) when
    scene-level evidence already supports it.
    """
    labels = {
        e.label.lower().strip()
        for e in entities
        if e.confidence >= 0.50 and e.narrative_safe
    }
    st = (scene_type or "").lower().strip()
    se = (setting or "").lower().strip()
    conf = float(confidence)
    blob = (evidence_blob or "").lower()
    io = (indoor_outdoor or "").lower().strip()

    kitchen_cues = labels & {
        "oven",
        "refrigerator",
        "sink",
        "microwave",
        "toaster",
        "stove",
    }
    office_cues = labels & {"laptop", "keyboard", "mouse"}
    restaurant_service = labels & {"wine glass", "fork", "knife", "spoon"}
    vehicle_cues = labels & {"car", "bus", "truck", "motorcycle", "van", "taxi"}
    road_cues = labels & {"road", "traffic light", "stop sign", "parking meter", "fire hydrant"}
    nature_cues = labels & {
        "tree",
        "grass",
        "mountain",
        "sky",
        "bear",
        "bird",
        "horse",
        "dog",
        "cow",
        "sheep",
        "elephant",
        "zebra",
        "giraffe",
        "cat",
    }
    sport_equipment = labels & {
        "skis",
        "snowboard",
        "tennis racket",
        "skateboard",
        "surfboard",
        "sports ball",
        "baseball bat",
        "frisbee",
        "kite",
    }
    verified_acts = {
        a.activity.lower()
        for a in activities
        if a.qa_safe or (a.narrative_safe and a.confidence >= 0.75)
    }
    person_count = sum(
        1
        for e in entities
        if e.narrative_safe
        and e.label.lower() in {"person", "man", "woman", "child", "people"}
    )

    # Strong kitchen appliances dominate weak restaurant/office/classroom guesses.
    if kitchen_cues and st in {
        "restaurant",
        "cafe",
        "office",
        "classroom",
        "meeting room",
        "dining room",
        "dining area",
    }:
        return "kitchen", "kitchen", max(conf, 0.72)

    # Restaurant/cafe without serviceware + without kitchen → too weak; neutralize.
    if st in {"restaurant", "cafe"} and not restaurant_service and not kitchen_cues:
        if "dining table" in labels or "chair" in labels:
            return "indoor scene", "indoor room", min(conf, 0.55)

    # Office without computing devices → neutralize (chair/tv alone is not an office).
    if st in {"office", "office workspace"} and not office_cues:
        if kitchen_cues:
            return "kitchen", "kitchen", max(conf, 0.70)
        return "indoor scene", "indoor room", min(conf, 0.55)

    # Classroom without backpack + study materials → neutralize generic guesses.
    if st == "classroom":
        has_classroom = (
            "backpack" in labels
            and "book" in labels
            and labels & {"laptop", "keyboard", "chair"}
        )
        if not has_classroom:
            return "indoor scene", "indoor room", min(conf, 0.55)

    # Laboratory without fixture cluster (bottle+bowl+cup) → neutralize.
    if st in {"laboratory", "lab"}:
        if len(labels & {"bottle", "bowl", "cup"}) < 3:
            return "indoor scene", "indoor room", min(conf, 0.55)

    # Library without multi-cue reading furniture → neutralize (a book alone is not a library).
    if st == "library":
        if not ("book" in labels and len(labels & {"book", "chair", "laptop"}) >= 3):
            return "indoor scene", "indoor room", min(conf, 0.55)

    # Vehicle alone is not a highway — prefer broader urban/outdoor labels.
    if st in {"highway", "freeway", "motorway"}:
        if not road_cues:
            if vehicle_cues:
                return "urban environment", "outdoor area", min(conf, 0.58)
            return "outdoor scene", "outdoor area", min(conf, 0.55)

    # Sport equipment alone is not a named court/field venue.
    if st in {
        "tennis court",
        "football field",
        "basketball court",
        "skate park",
        "sports field",
    }:
        def _compatible_sport(venue: str, act: str) -> bool:
            if venue == "tennis court":
                # "playing with a tennis racket" is possession/use, not court play.
                return "playing tennis" in act or act.startswith("playing tennis")
            if venue == "basketball court":
                return "basketball" in act
            if venue in {"football field", "sports field"}:
                return any(tok in act for tok in ("football", "soccer", "baseball"))
            if venue == "skate park":
                return "skate" in act
            return False

        has_compatible = any(_compatible_sport(st, a) for a in verified_acts)
        # Need a venue-compatible verified activity — else broaden.
        if not has_compatible:
            if sport_equipment and person_count >= 1:
                return "outdoor scene", "recreational area", min(conf, 0.58)
            return "outdoor scene", "outdoor area", min(conf, 0.55)

    # Bicycle roadside without road infrastructure → outdoor scene.
    # Never invent trail from bicycle alone (forbidden object→venue shortcut).
    if st in {"road", "roadside"} and "bicycle" in labels and not road_cues and not vehicle_cues - {"bicycle"}:
        if nature_cues:
            return "natural environment", "outdoor area", min(max(conf, 0.55), 0.65)
        return "outdoor scene", "outdoor area", min(conf, 0.58)

    # Farm/pasture needs more than one livestock label (horse alone ≠ farm).
    if st in {"farm", "farm pasture", "pasture"} or "farm" in se:
        livestock = labels & {"cow", "horse", "sheep", "goat", "chicken"}
        farm_structure = labels & {"fence", "barn", "tractor"}
        if len(livestock) < 2 and not (livestock and farm_structure):
            if nature_cues:
                return "natural environment", "outdoor area", min(conf, 0.58)
            return "outdoor scene", "outdoor area", min(conf, 0.55)

    # Office requires at least two computing cues — one laptop alone is weak.
    if st in {"office", "office workspace"} and len(office_cues) < 2:
        if kitchen_cues:
            return "kitchen", "kitchen", max(conf, 0.70)
        return "indoor scene", "indoor room", min(conf, 0.55)

    # Restaurant needs serviceware + table; one utensil is insufficient after table alone.
    if st in {"restaurant", "cafe"} and "dining table" in labels:
        if len(restaurant_service) < 1 and not kitchen_cues:
            return "indoor scene", "indoor room", min(conf, 0.55)

    # Prefer sports/outdoor cues when present over generic indoor.
    if sport_equipment and st in {"indoor scene", "unknown", "office", "classroom"}:
        return (
            "outdoor scene" if st in {"office", "classroom", "unknown"} else st,
            se or "outdoor area",
            conf,
        )

    # If a verified activity names a concrete place-compatible action, keep current label.
    _ = verified_acts
    if not st or st in {"unknown", "general"}:
        if kitchen_cues:
            return "kitchen", "kitchen", max(conf, 0.70)
        if len(office_cues) >= 2:
            return "office", "office workspace", max(conf, 0.68)
        if nature_cues and not kitchen_cues and not office_cues:
            # Wildlife / nature objects without indoor furniture → natural outdoor.
            indoor_furniture = labels & {
                "chair",
                "couch",
                "bed",
                "dining table",
                "tv",
                "refrigerator",
                "oven",
                "sink",
            }
            if not indoor_furniture:
                return "natural environment", "outdoor area", min(max(conf, 0.55), 0.65)
        if vehicle_cues and not kitchen_cues:
            return "urban environment", "outdoor area", min(max(conf, 0.52), 0.60)
        indoor = any(
            lab
            in {
                "chair",
                "couch",
                "bed",
                "dining table",
                "tv",
                "refrigerator",
                "oven",
                "sink",
            }
            for lab in labels
        )
        if indoor:
            return "indoor scene", "indoor room", min(max(conf, 0.50), 0.60)

    out_type, out_setting, out_conf = scene_type or st, setting or se, conf

    # Preserve explicit trail/path scene evidence — do not collapse to "outdoor area"
    # when VLM/scene text already names a trail and road infrastructure is absent.
    # Hierarchy: strong scene text > object shortcuts (bicycle alone must NOT force trail).
    trail_mentioned = bool(
        re.search(r"\b(?:dirt\s+)?trail\b|\bfootpath\b|\bhiking\s+path\b|\bdirt\s+path\b", blob)
    )
    urban_road_mentioned = bool(
        re.search(r"\b(?:highway|freeway|motorway|city street|crosswalk|parking lot)\b", blob)
    )
    already_trail = out_setting in {"trail", "outdoor trail", "dirt trail", "mountain trail"}
    outdoorish = io == "outdoor" or out_type in {
        "outdoor scene",
        "natural environment",
        "urban environment",
        "park",
        "mountain",
        "forest",
        "",
        "unknown",
        "general",
    }
    strong_urban = bool(road_cues) or bool(vehicle_cues - {"bicycle", "motorcycle"}) or urban_road_mentioned
    if outdoorish and (trail_mentioned or already_trail) and not strong_urban:
        # Mountain trail only when mountain evidence is actually present.
        if (
            "mountain trail" in blob
            or out_setting == "mountain trail"
            or ("mountain" in blob and trail_mentioned)
        ) and ("mountain" in labels or "mountain" in blob):
            return "natural environment", "mountain trail", max(out_conf, 0.68)
        return "natural environment", "outdoor trail", max(out_conf, 0.68)

    # If we somehow carried a trail setting into a strong urban road scene, broaden.
    if already_trail and strong_urban and not trail_mentioned:
        return "urban environment", "outdoor area", min(out_conf, 0.58)

    return out_type, out_setting, out_conf


_RELATION_VERBS = frozenset(
    {
        "holding",
        "talking_to",
        "looking_at",
        "using",
        "playing",
        "touching",
        "interacting",
        "wearing",
        "sitting_on",
        "riding",
        "leading",
        "carrying",
        "near",
        "beside",
    }
)


def _brief_introduces_unverified_relations(brief: str, verified: VerifiedSceneEvidence) -> bool:
    lower = (brief or "").lower()
    allowed = {r.relation_type.lower() for r in verified.relations if r.narrative_safe or r.qa_safe}
    for verb in _RELATION_VERBS:
        if verb.replace("_", " ") in lower or verb in lower:
            if verb not in allowed and verb.replace(" ", "_") not in allowed:
                return True
    return False


def language_understanding_from_verified(
    verified: VerifiedSceneEvidence,
    base: SceneUnderstanding | None = None,
) -> SceneUnderstanding:
    """Project verified evidence into SceneUnderstanding for caption writers.

    Caption paths that still accept SceneUnderstanding must receive this projection,
    not the raw reasoner output, so weak relations cannot bypass the gate.
    """
    facts: list[EvidenceFact] = []
    for ent in verified.entities:
        if not ent.narrative_safe:
            continue
        subject = f"{ent.label} #{ent.object_index + 1}"
        facts.append(
            EvidenceFact(subject, "is", ent.label, ent.confidence, ent.source)
        )
    for attr in verified.narrative_attributes():
        ent = verified.entity_by_id(attr.entity_id)
        subject = (
            f"{ent.label} #{ent.object_index + 1}" if ent is not None else attr.entity_id
        )
        facts.append(
            EvidenceFact(subject, attr.name, attr.value, attr.confidence, attr.source)
        )
    for rel in verified.narrative_relations():
        sub = verified.entity_by_id(rel.subject_id)
        obj = verified.entity_by_id(rel.object_id)
        sub_s = (
            f"{sub.label} #{sub.object_index + 1}" if sub is not None else rel.subject_id
        )
        obj_s = (
            f"{obj.label} #{obj.object_index + 1}" if obj is not None else rel.object_id
        )
        facts.append(
            EvidenceFact(sub_s, rel.relation_type, obj_s, rel.confidence, rel.source)
        )
    for act in verified.activities:
        # Caption activity facts: CONFIRMED + narrative_safe only.
        # SUPPORTED actions are QA-only so caption generation stays unchanged.
        if act.evidence_level != ActivityEvidenceLevel.CONFIRMED:
            continue
        if not act.narrative_safe or not act.qa_safe:
            continue
        subject = "scene"
        if act.entity_ids:
            ent = verified.entity_by_id(act.entity_ids[0])
            if ent is not None:
                subject = f"{ent.label} #{ent.object_index + 1}"
        facts.append(
            EvidenceFact(subject, "activity", act.activity, act.confidence, act.source)
        )
    if verified.scene.indoor_outdoor:
        facts.append(
            EvidenceFact(
                "scene",
                "indoor_outdoor",
                verified.scene.indoor_outdoor,
                verified.scene.confidence,
                "environment",
            )
        )
    if verified.scene.setting:
        facts.append(
            EvidenceFact(
                "scene",
                "setting",
                verified.scene.setting,
                verified.scene.confidence,
                "environment",
            )
        )

    ranked: list[str] = []
    for eid in verified.ranked_entity_ids:
        ent = verified.entity_by_id(eid)
        if ent is not None:
            ranked.append(f"{ent.label} #{ent.object_index + 1}")
    if not ranked:
        ranked = [
            f"{e.label} #{e.object_index + 1}"
            for e in verified.entities
            if e.narrative_safe
        ]

    env_keys = tuple(
        k
        for k in (
            verified.scene.indoor_outdoor,
            verified.scene.setting,
            verified.scene.scene_type,
            verified.scene.time_of_day,
            verified.scene.weather,
        )
        if k
    )
    # Preserve verified hazard facts from the reasoner base (fire/smoke are not COCO entities).
    if base is not None:
        for fact in base.facts:
            is_hazard = (
                (fact.predicate == "hazard" and fact.value.lower() in {"fire", "smoke"})
                or (
                    fact.subject.lower() in {"fire", "smoke"}
                    and fact.predicate in {"is", "detected", "present", "visible"}
                )
            )
            if is_hazard and fact.confidence >= 0.60:
                facts.append(fact)
                key = f"hazard={fact.value.lower() if fact.predicate == 'hazard' else fact.subject.lower()}"
                if key not in env_keys:
                    env_keys = (*env_keys, key)
        for line in getattr(verified.scene, "evidence", ()) or ():
            low = (line or "").lower()
            for lab in ("fire", "smoke"):
                key = f"hazard={lab}"
                if lab in low and key not in env_keys:
                    env_keys = (*env_keys, key)
    activity_keys = tuple(a.activity for a in verified.activities if a.narrative_safe)
    discarded = len(verified.rejected)
    if base is not None:
        discarded = max(discarded, base.discarded_count)
    return SceneUnderstanding(
        facts=tuple(facts),
        ranked_subjects=tuple(ranked),
        environment_keys=env_keys,
        activity_keys=activity_keys,
        ocr_text=verified.ocr_text,
        evidence_brief=verified.as_evidence_brief(),
        overall_confidence=verified.overall_confidence,
        discarded_count=discarded,
        contradictions_resolved=base.contradictions_resolved if base is not None else 0,
    )
