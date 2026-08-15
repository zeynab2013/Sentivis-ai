"""Controlled VLM → activity bridge.

VLM prose may propose activities, but they become ActivityEvidence only when
independent visual evidence corroborates them. Never upgrades possession to
venue sports (racket ≠ tennis, cup ≠ drinking, near bike ≠ riding).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.contracts.analysis import ActivityEvidence, SceneContext, SceneGraph

_PERSON = frozenset({"person", "people", "man", "woman", "child"})


def _largest_person_indices(
    graph: SceneGraph,
    person_indices: list[int],
    *,
    limit: int = 2,
) -> list[int]:
    """Return person node indices ranked by area (largest first), capped at limit."""
    by_idx = {n.index: n for n in graph.nodes}
    ranked = sorted(
        person_indices,
        key=lambda i: float(getattr(by_idx.get(i), "bounding_box_area_ratio", 0.0) or 0.0),
        reverse=True,
    )
    return ranked[: max(0, int(limit))]


def _shared_sport_support_nodes(
    graph: SceneGraph,
    person_indices: list[int],
    equip_idx: list[int],
    *,
    seed_person: int | None = None,
) -> tuple[int, ...]:
    """Bind a shared sport activity to up to two strongest people + equipment.

    Does not attach every person in the scene — only the largest relevant actors
    (preferring an interaction seed when present).
    """
    if not person_indices:
        return tuple(equip_idx[:1])
    largest = _largest_person_indices(graph, person_indices, limit=2)
    actors: list[int] = []
    if seed_person is not None and seed_person in person_indices:
        actors.append(seed_person)
    for pi in largest:
        if pi not in actors:
            actors.append(pi)
        if len(actors) >= 2:
            break
    if len(person_indices) >= 2 and len(actors) < 2:
        for pi in person_indices:
            if pi not in actors:
                actors.append(pi)
            if len(actors) >= 2:
                break
    nodes = list(actors[:2] if len(person_indices) >= 2 else actors[:1])
    if equip_idx:
        ball = equip_idx[0]
        if ball not in nodes:
            nodes.append(ball)
    return tuple(nodes)

# Literal interaction patterns extracted from VLM text (object required).
_LITERAL_PATTERNS: tuple[tuple[re.Pattern[str], str, frozenset[str], frozenset[str]], ...] = (
    # (pattern, activity_phrase, required_object_labels, required_relation_types)
    (
        re.compile(r"\briding (?:a |an |the )?dirt bike\b", re.I),
        "riding a motorcycle",
        frozenset({"motorcycle"}),
        frozenset({"riding"}),
    ),
    (
        re.compile(r"\briding (?:a |an |the )?motorcycle\b", re.I),
        "riding a motorcycle",
        frozenset({"motorcycle"}),
        frozenset({"riding"}),
    ),
    (
        re.compile(r"\briding (?:a |an |the )?(?:bi)?cycle\b", re.I),
        "riding a bicycle",
        frozenset({"bicycle"}),
        frozenset({"riding"}),
    ),
    (
        # Generic "bike" → bicycle when bicycle present; motorcycle only if no bicycle.
        re.compile(r"\briding (?:a |an |the )?bike\b", re.I),
        "riding a bicycle",
        frozenset({"bicycle"}),
        frozenset({"riding"}),
    ),
    (
        re.compile(r"\briding (?:a |an |the )?horse\b", re.I),
        "riding a horse",
        frozenset({"horse"}),
        frozenset({"riding"}),
    ),
    (
        re.compile(r"\bleading (?:a |an |the )?horse\b", re.I),
        "leading a horse",
        frozenset({"horse"}),
        frozenset({"leading", "guiding"}),
    ),
    (
        re.compile(r"\bholding (?:a |an |the )?rope\b", re.I),
        "holding a rope",
        frozenset({"horse"}),  # rope often unlabeled; horse+holding/leading corroborates
        frozenset({"holding", "leading", "guiding", "carrying"}),
    ),
    (
        re.compile(r"\b(?:holding|swinging) (?:a |an |the )?(?:baseball )?bat\b", re.I),
        "holding a baseball bat",
        frozenset({"baseball bat"}),
        frozenset({"holding", "using", "playing_with", "carrying"}),
    ),
    (
        re.compile(r"\b(?:holding|playing with) (?:a |an |the )?(?:tennis )?racket\b", re.I),
        "holding a tennis racket",
        frozenset({"tennis racket"}),
        frozenset({"holding", "using", "playing_with", "carrying"}),
    ),
    (
        re.compile(r"\bskateboarding\b|\bridging (?:a |an |the )?skateboard\b", re.I),
        "skateboarding",
        frozenset({"skateboard"}),
        frozenset({"riding", "playing_with", "using", "holding"}),
    ),
    (
        re.compile(
            r"\b(?:playing with|kicking|chasing|controlling) (?:a |an |the )?(?:soccer |football )?ball\b"
            r"|\b(?:soccer|football) (?:game|match|player|players)\b"
            r"|\bplayers? (?:are |is )?(?:playing|kicking)\b",
            re.I,
        ),
        "playing with a ball",
        frozenset({"sports ball"}),
        frozenset({"playing_with", "holding", "using", "kicking"}),
    ),
)

# Named sport only when VLM names it AND multi-signal object+interaction evidence exists.
_SPORT_PATTERNS: tuple[tuple[re.Pattern[str], str, frozenset[str]], ...] = (
    (
        re.compile(r"\b(?:playing |play )?(?:football|soccer)\b", re.I),
        "playing football",
        frozenset({"sports ball"}),
    ),
    (
        re.compile(r"\b(?:playing |play )?baseball\b|\bswinging (?:a |an |the )?(?:baseball )?bat\b", re.I),
        "playing baseball",
        frozenset({"baseball bat", "sports ball"}),
    ),
)


@dataclass(frozen=True)
class VlmActivityCandidate:
    activity: str
    confidence: float
    node_indices: tuple[int, ...]
    relation_types: tuple[str, ...]
    rationale: str


def extract_vlm_activity_candidates(
    vlm_text: str,
    scene_context: SceneContext,
    *,
    vlm_confidence: float = 0.75,
) -> tuple[ActivityEvidence, ...]:
    """Return ActivityEvidence proposals corroborated by graph entities/relations."""
    text = (vlm_text or "").strip()
    if not text:
        return ()

    graph = scene_context.graph
    labels = {n.index: (n.label or "").lower() for n in graph.nodes}
    label_set = set(labels.values())
    person_indices = [i for i, lab in labels.items() if lab in _PERSON]
    if not person_indices:
        return ()

    rel_types_by_pair: dict[tuple[int, int], set[str]] = {}
    for rel in graph.relations:
        key = (rel.subject_index, rel.object_index)
        rel_types_by_pair.setdefault(key, set()).add(rel.relation_type.lower())
        rel_types_by_pair.setdefault((rel.object_index, rel.subject_index), set()).add(
            rel.relation_type.lower()
        )

    def _has_relation(person_i: int, obj_i: int, allowed: frozenset[str]) -> bool:
        types = rel_types_by_pair.get((person_i, obj_i), set())
        return bool(types & set(allowed))

    def _object_indices(wanted: frozenset[str]) -> list[int]:
        return [i for i, lab in labels.items() if lab in wanted]

    conf_base = max(0.55, min(0.88, float(vlm_confidence) * 0.95))
    out: list[ActivityEvidence] = []
    seen: set[str] = set()

    for pattern, phrase, obj_labels, rel_need in _LITERAL_PATTERNS:
        if not pattern.search(text):
            continue
        # Holding rope: allow leading horse as corroboration when rope label absent.
        objs = _object_indices(obj_labels - _PERSON)
        if phrase == "holding a rope" and not objs:
            objs = _object_indices(frozenset({"horse"}))
            rel_need_eff = frozenset({"holding", "leading", "guiding", "carrying"})
        else:
            rel_need_eff = rel_need
        if not objs and not (obj_labels & label_set):
            continue
        # Prefer person–object pairs with required relation.
        matched = False
        for pi in person_indices[:3]:
            for oi in objs[:3]:
                if _has_relation(pi, oi, rel_need_eff):
                    key = phrase
                    if key in seen:
                        matched = True
                        break
                    seen.add(key)
                    support = tuple(
                        sorted(rel_types_by_pair.get((pi, oi), set()) & set(rel_need_eff))
                    ) or tuple(sorted(rel_need_eff)[:1])
                    out.append(
                        ActivityEvidence(
                            activity=phrase,
                            confidence=conf_base,
                            supporting_node_indices=(pi, oi),
                            supporting_relation_types=support,
                            rationale=f"vlm_literal+relation:{support[0] if support else 'n/a'}",
                        )
                    )
                    matched = True
                    break
            if matched:
                break
        # Riding: also accept when riding relation exists to the vehicle label family.
        if not matched and phrase.startswith("riding "):
            for rel in graph.relations:
                if rel.relation_type.lower() != "riding":
                    continue
                sub_l = labels.get(rel.subject_index, "")
                obj_l = labels.get(rel.object_index, "")
                if sub_l in _PERSON and obj_l in obj_labels:
                    if phrase in seen:
                        break
                    seen.add(phrase)
                    out.append(
                        ActivityEvidence(
                            activity=phrase,
                            confidence=max(conf_base, rel.confidence),
                            supporting_node_indices=(rel.subject_index, rel.object_index),
                            supporting_relation_types=("riding",),
                            rationale="vlm_literal+riding_relation",
                        )
                    )
                    break

    # Named sports: VLM names sport + equipment. Prefer geometric interaction;
    # for football/soccer, allow multi-signal without IoU when VLM+ball+people+field.
    for pattern, phrase, equip in _SPORT_PATTERNS:
        if not pattern.search(text):
            continue
        if not (equip & label_set):
            continue
        if len(person_indices) < 1:
            continue
        equip_idx = _object_indices(equip)
        interactive = False
        support_rel = ""
        pair: tuple[int, int] | None = None
        for pi in person_indices[:4]:
            for oi in equip_idx[:4]:
                types = rel_types_by_pair.get((pi, oi), set())
                hit = types & {"playing_with", "holding", "using", "carrying"}
                if hit:
                    interactive = True
                    support_rel = sorted(hit)[0]
                    pair = (pi, oi)
                    break
            if interactive:
                break

        fieldish = bool(re.search(r"\b(?:field|pitch|stadium|grass)\b", text, re.I))
        multi_signal_football = (
            phrase == "playing football"
            and "sports ball" in label_set
            and len(person_indices) >= 2
            and fieldish
        )

        support_nodes: tuple[int, ...] | None = None
        if not interactive or pair is None:
            if not multi_signal_football:
                continue
            # Multi-signal without bbox contact: VLM named sport + ball + people + field.
            # Bind the two largest people + ball — not person_indices[0] alone.
            support_nodes = _shared_sport_support_nodes(
                graph, person_indices, equip_idx
            )
            support_rel = "vlm_multisignal"
        else:
            support_nodes = pair
            # Shared football with ≥2 people: expand ownership beyond the single
            # interacting person so co-players are activity actors.
            if phrase == "playing football" and len(person_indices) >= 2:
                support_nodes = _shared_sport_support_nodes(
                    graph,
                    person_indices,
                    equip_idx,
                    seed_person=pair[0],
                )
        # Multi-signal gate for named football/soccer: need ball + ≥2 people OR field cue.
        if phrase == "playing football":
            if len(person_indices) < 2 and not fieldish:
                continue
            if "sports ball" not in label_set:
                continue
        if phrase == "playing baseball":
            # Prefer bat interaction; ball alone is not enough for the named sport.
            if "baseball bat" not in label_set:
                continue
            if not interactive:
                continue
        if phrase in seen:
            continue
        seen.add(phrase)
        # Keep literal when only possession (holding) — named sport only with playing_with/using
        # OR VLM swinging + bat holding (baseball).
        if support_rel == "holding" and phrase == "playing football":
            # Holding ball alone → literal playing with a ball, not named sport upgrade.
            lit = "playing with a ball"
            if lit not in seen:
                seen.add(lit)
                out.append(
                    ActivityEvidence(
                        activity=lit,
                        confidence=min(conf_base, 0.72),
                        supporting_node_indices=pair or support_nodes or (),
                        supporting_relation_types=(support_rel,),
                        rationale="vlm_sport_softened_to_literal_ball",
                    )
                )
            continue
        if support_rel == "holding" and phrase == "playing baseball":
            # Swinging/playing baseball in VLM + holding bat → keep holding bat (literal CONFIRMED path).
            lit = "holding a baseball bat"
            if lit not in seen:
                seen.add(lit)
                out.append(
                    ActivityEvidence(
                        activity=lit,
                        confidence=conf_base,
                        supporting_node_indices=pair or support_nodes or (),
                        supporting_relation_types=("holding",),
                        rationale="vlm_baseball+holding_bat",
                    )
                )
            continue
        out.append(
            ActivityEvidence(
                activity=phrase,
                confidence=min(conf_base, 0.78 if support_rel != "vlm_multisignal" else 0.74),
                supporting_node_indices=support_nodes or (),
                supporting_relation_types=(support_rel,),
                rationale=(
                    "vlm_named_sport+multisignal"
                    if support_rel == "vlm_multisignal"
                    else f"vlm_named_sport+{support_rel}"
                ),
            )
        )

    return tuple(out)


def merge_vlm_activities_into_context(
    scene_context: SceneContext,
    vlm_text: str,
    *,
    vlm_confidence: float = 0.75,
) -> SceneContext:
    """Return a shallow-copied context with corroborated VLM activities appended."""
    extras = extract_vlm_activity_candidates(
        vlm_text, scene_context, vlm_confidence=vlm_confidence
    )
    if not extras:
        return scene_context
    existing = list(scene_context.activities.activities)
    existing_text = {a.activity.lower().strip() for a in existing}
    for item in extras:
        key = item.activity.lower().strip()
        if key in existing_text:
            continue
        existing.append(item)
        existing_text.add(key)
    from core.contracts.analysis import ActivityHints

    new_hints = ActivityHints(
        activities=tuple(existing),
        confidence=max(scene_context.activities.confidence, max(a.confidence for a in extras)),
    )
    return SceneContext(
        graph=scene_context.graph,
        attributes=scene_context.attributes,
        activities=new_hints,
        environment=scene_context.environment,
        object_count=scene_context.object_count,
        dominant_objects=scene_context.dominant_objects,
        spatial_summary=scene_context.spatial_summary,
    )
