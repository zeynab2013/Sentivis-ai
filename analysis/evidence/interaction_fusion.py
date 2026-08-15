"""Multi-signal interaction evidence fusion.

VLM prose is a candidate only. Geometry / mask / pose corroboration is required
before an interaction enters VerifiedSceneEvidence.

Architecture:
  detections + spatial + attributes + VLM candidates + pose
  → InteractionEvidenceFuser
  → fused Relation edges (source-traced)
  → SceneContext / VerifiedSceneEvidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from analysis.common.geometry import (
    euclidean_distance,
    image_diagonal,
    intersection_over_union,
)
from analysis.common.mask_geometry import mask_overlap_ratio
from analysis.pose.pose_estimator import PoseEstimate
from core.contracts.analysis import Relation, SceneContext, SceneGraph, SceneNode
from core.contracts.detection import Detection, DetectionResult
from core.contracts.language import VisualObservations
from core.logging import get_logger

logger = get_logger(__name__)

_PERSON = frozenset({"person", "people", "man", "woman", "child", "skier", "rider"})
_HOLDABLE = frozenset(
    {
        "cup",
        "bottle",
        "book",
        "cell phone",
        "phone",
        "laptop",
        "wine glass",
        "fork",
        "knife",
        "spoon",
        "umbrella",
        "remote",
        "toothbrush",
    }
)
_RIDEABLE = frozenset({"horse", "bicycle", "motorcycle", "elephant", "bike"})
_USABLE = frozenset({"laptop", "keyboard", "mouse", "remote", "cell phone", "phone"})
_FOOD_DRINK = frozenset(
    {"cup", "bottle", "wine glass", "bowl", "banana", "apple", "sandwich", "pizza", "cake", "donut", "hot dog"}
)
_PLAYABLE = frozenset(
    {"sports ball", "tennis racket", "skateboard", "frisbee", "kite", "baseball bat", "teddy bear"}
)

# Interaction verbs we may recover / boost (never from proximity alone).
_INTERACTION_PATTERNS: tuple[tuple[str, re.Pattern[str], frozenset[str]], ...] = (
    (
        "holding",
        re.compile(
            r"\b(?:hold(?:s|ing)?|held|grasping|clutching)\s+(?:a |an |the )?([\w\s\-]+)",
            re.I,
        ),
        _HOLDABLE,
    ),
    (
        "carrying",
        re.compile(
            r"\b(?:carry(?:s|ing)?|carried)\s+(?:a |an |the )?([\w\s\-]+)",
            re.I,
        ),
        frozenset({"handbag", "backpack", "suitcase", "umbrella", "sports ball"}),
    ),
    (
        "using",
        re.compile(
            r"\b(?:us(?:es|ing)|operat(?:es|ing)|typ(?:es|ing) on)\s+(?:a |an |the )?([\w\s\-]+)",
            re.I,
        ),
        _USABLE,
    ),
    (
        "riding",
        re.compile(
            r"\b(?:rid(?:es|ing)|mounted on)\s+(?:a |an |the )?([\w\s\-]+)",
            re.I,
        ),
        _RIDEABLE,
    ),
    (
        "playing_with",
        re.compile(
            r"\b(?:play(?:s|ing)(?: with)?)\s+(?:a |an |the )?([\w\s\-]+)",
            re.I,
        ),
        _PLAYABLE,
    ),
    (
        "eating",
        re.compile(
            r"\b(?:eat(?:s|ing)|chewing)\s+(?:a |an |the )?([\w\s\-]+)?",
            re.I,
        ),
        _FOOD_DRINK,
    ),
    (
        "drinking",
        re.compile(
            r"\b(?:drink(?:s|ing)|sipping)\s+(?:from )?(?:a |an |the )?([\w\s\-]+)?",
            re.I,
        ),
        frozenset({"cup", "bottle", "wine glass", "bowl"}),
    ),
)

# Speculative — VLM alone never verifies these.
_SPECULATIVE = frozenset({"talking_to", "looking_at", "interacting_with"})
_TALK_PATTERN = re.compile(
    r"\b(?:talk(?:s|ing)?(?: to)?|convers(?:e|ing)|speaking(?: with| to)?|chatting)\b",
    re.I,
)
_LOOK_PATTERN = re.compile(r"\b(?:look(?:s|ing)? at|gazing at|staring at)\b", re.I)

_LABEL_ALIASES = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "bike": "bicycle",
    "cycle": "bicycle",
    "computer": "laptop",
    "notebook": "laptop",
}


@dataclass(frozen=True)
class SpatialSupport:
    """Geometric corroboration for a person–object pair."""

    proximity: float
    iou: float
    mask_overlap: float
    hand_zone: bool
    in_front: bool
    above_object: bool  # person center above object (riding/sitting cues)


@dataclass(frozen=True)
class VlmRelationCandidate:
    """Parsed VLM claim — not yet a verified fact."""

    relation_type: str
    object_phrase: str
    raw_span: str
    confidence: float


@dataclass(frozen=True)
class InteractionEvidenceRecord:
    """Audit trail for one fused interaction decision."""

    subject_index: int
    object_index: int
    relation_type: str
    confidence: float
    accepted: bool
    sources: tuple[str, ...]
    reason: str
    spatial: SpatialSupport | None = None


@dataclass(frozen=True)
class InteractionFusionResult:
    """Fused relations plus audit records."""

    relations: tuple[Relation, ...]
    records: tuple[InteractionEvidenceRecord, ...]
    boosted: int
    recovered: int
    rejected_vlm: int


def _normalize_label(text: str) -> str:
    token = (text or "").strip().lower()
    token = re.sub(r"[^a-z0-9\s\-]", "", token).strip()
    # Keep first 1–3 words for phrase matching.
    parts = token.split()
    for n in (3, 2, 1):
        if len(parts) >= n:
            candidate = " ".join(parts[:n])
            return _LABEL_ALIASES.get(candidate, candidate)
    return _LABEL_ALIASES.get(token, token)


def extract_vlm_relation_candidates(
    text: str,
    *,
    vlm_confidence: float = 0.6,
) -> tuple[VlmRelationCandidate, ...]:
    """Parse free-form VLM caption into interaction candidates (not facts)."""
    body = (text or "").strip()
    if not body:
        return ()
    conf = max(0.35, min(0.85, float(vlm_confidence) * 0.9))
    found: list[VlmRelationCandidate] = []
    seen: set[tuple[str, str]] = set()

    for relation_type, pattern, _allowed in _INTERACTION_PATTERNS:
        for match in pattern.finditer(body):
            phrase = (match.group(1) or "").strip()
            key = (relation_type, _normalize_label(phrase) if phrase else "")
            if key in seen:
                continue
            seen.add(key)
            found.append(
                VlmRelationCandidate(
                    relation_type=relation_type,
                    object_phrase=phrase,
                    raw_span=match.group(0),
                    confidence=conf,
                )
            )

    # Speculative candidates recorded for rejection audit only.
    if _TALK_PATTERN.search(body):
        found.append(
            VlmRelationCandidate("talking_to", "person", "talking", conf * 0.7)
        )
    if _LOOK_PATTERN.search(body):
        found.append(
            VlmRelationCandidate("looking_at", "object", "looking", conf * 0.7)
        )
    return tuple(found)


def compute_spatial_support(
    person: Detection,
    obj: Detection,
    *,
    diagonal: float,
) -> SpatialSupport:
    """Compute spatial cues without emitting a relation."""
    pbox = person.bounding_box
    obox = obj.bounding_box
    distance = euclidean_distance(pbox, obox)
    radius = max(diagonal * 0.28, 1.0)
    proximity = 1.0 - min(1.0, distance / radius)
    iou = intersection_over_union(pbox, obox)
    overlap = 0.0
    if person.segmentation is not None and obj.segmentation is not None:
        image_area = float(max(1.0, pbox.area + obox.area))
        overlap = mask_overlap_ratio(person.segmentation, obj.segmentation, image_area)
    hand_zone = (
        obox.center_y >= pbox.center_y - pbox.height * 0.15
        and obox.center_y <= pbox.y_max + pbox.height * 0.05
        and abs(obox.center_x - pbox.center_x) <= pbox.width * 0.75
    )
    in_front = (
        abs(obox.center_x - pbox.center_x) < max(pbox.width, obox.width) * 0.85
        and obox.center_y >= pbox.center_y - pbox.height * 0.15
    )
    above_object = pbox.center_y < obox.center_y and iou >= 0.02
    return SpatialSupport(
        proximity=proximity,
        iou=iou,
        mask_overlap=overlap,
        hand_zone=hand_zone,
        in_front=in_front,
        above_object=above_object,
    )


def _contact_ok(spatial: SpatialSupport, relation_type: str) -> bool:
    """True when spatial evidence supports contact/use — not mere nearness."""
    if relation_type == "holding":
        return (spatial.iou >= 0.04 or spatial.mask_overlap >= 0.03) and (
            spatial.hand_zone or spatial.iou >= 0.10
        )
    if relation_type == "carrying":
        return (spatial.iou >= 0.04 or spatial.mask_overlap >= 0.03) and (
            spatial.hand_zone or spatial.proximity >= 0.70
        )
    if relation_type == "using":
        return spatial.in_front and (spatial.iou >= 0.02 or spatial.proximity >= 0.65)
    if relation_type == "riding":
        return spatial.above_object or spatial.iou >= 0.08
    if relation_type in {"eating", "drinking"}:
        return spatial.hand_zone and (spatial.iou >= 0.03 or spatial.proximity >= 0.70)
    if relation_type == "playing_with":
        return spatial.proximity >= 0.70 and (spatial.iou >= 0.03 or spatial.hand_zone)
    return False


def _label_match(phrase: str, label: str) -> bool:
    a = _normalize_label(phrase)
    b = _normalize_label(label)
    if not a:
        return True  # verb without object ("eating") — defer to class allowlist
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


class InteractionEvidenceFuser:
    """Fuse geometry + VLM candidates + pose into verified-grade relations."""

    def fuse(
        self,
        scene_context: SceneContext,
        *,
        detections: DetectionResult | None = None,
        observations: VisualObservations | None = None,
        poses: tuple[PoseEstimate, ...] | None = None,
    ) -> InteractionFusionResult:
        graph = scene_context.graph
        nodes = list(graph.nodes)
        existing = list(graph.relations)
        by_key: dict[tuple[int, int, str], Relation] = {
            (r.subject_index, r.object_index, r.relation_type): r for r in existing
        }

        vlm_text = ""
        vlm_conf = 0.55
        if observations is not None and observations.raw_caption is not None:
            vlm_text = (observations.raw_caption.text or "").strip()
            vlm_conf = float(observations.confidence or observations.raw_caption.confidence or 0.55)
            for hint in observations.observations:
                if hint:
                    vlm_text += " " + hint
            for hint in observations.candidate_descriptions:
                if hint:
                    vlm_text += " " + hint

        candidates = extract_vlm_relation_candidates(vlm_text, vlm_confidence=vlm_conf)
        pose_by_index = {p.object_index: p for p in (poses or ())}

        det_by_index: dict[int, Detection] = {}
        diagonal = 1000.0
        if detections is not None:
            for i, det in enumerate(detections.detections):
                det_by_index[i] = det
            diagonal = image_diagonal(detections)

        records: list[InteractionEvidenceRecord] = []
        boosted = 0
        recovered = 0
        rejected_vlm = 0

        # --- Pass 1: boost existing interaction edges with VLM/pose agreement ---
        for key, rel in list(by_key.items()):
            if rel.relation_type in {"near", "next_to", "left_of", "right_of", "above", "below", "far"}:
                continue
            if rel.relation_type in _SPECULATIVE:
                continue
            sources = ["geometry"]
            spatial = self._spatial_for_indices(
                rel.subject_index, rel.object_index, nodes, det_by_index, diagonal
            )
            if spatial is not None and _contact_ok(spatial, rel.relation_type):
                sources.append("spatial")
            sub_label = self._label_at(nodes, rel.subject_index)
            obj_label = self._label_at(nodes, rel.object_index)
            vlm_hit = any(
                c.relation_type == rel.relation_type
                and (
                    _label_match(c.object_phrase, obj_label)
                    or (not c.object_phrase and rel.relation_type in {"eating", "drinking"})
                )
                for c in candidates
            )
            if vlm_hit:
                sources.append("vlm")
            pose = pose_by_index.get(rel.subject_index)
            if pose is not None and pose.action.replace(" ", "_") == rel.relation_type:
                sources.append("pose")
            if len(sources) >= 2:
                new_conf = min(0.95, rel.confidence + 0.08 * (len(sources) - 1))
                if new_conf > rel.confidence + 0.01:
                    boosted += 1
                by_key[key] = Relation(
                    rel.subject_index, rel.object_index, rel.relation_type, new_conf
                )
                records.append(
                    InteractionEvidenceRecord(
                        rel.subject_index,
                        rel.object_index,
                        rel.relation_type,
                        new_conf,
                        True,
                        tuple(sources),
                        "boosted_multi_signal",
                        spatial,
                    )
                )

        # --- Pass 2: recover interactions from VLM + spatial (not VLM alone) ---
        person_indices = [n.index for n in nodes if n.label.lower() in _PERSON]
        for cand in candidates:
            if cand.relation_type in _SPECULATIVE:
                rejected_vlm += 1
                records.append(
                    InteractionEvidenceRecord(
                        -1,
                        -1,
                        cand.relation_type,
                        cand.confidence,
                        False,
                        ("vlm",),
                        "speculative_requires_pose_gaze",
                    )
                )
                continue

            grounded = self._ground_candidate(
                cand, person_indices, nodes, det_by_index, diagonal
            )
            if grounded is None:
                rejected_vlm += 1
                records.append(
                    InteractionEvidenceRecord(
                        -1,
                        -1,
                        cand.relation_type,
                        cand.confidence,
                        False,
                        ("vlm",),
                        "entity_grounding_failed_or_ambiguous",
                    )
                )
                continue

            sub_i, obj_i, spatial = grounded
            key = (sub_i, obj_i, cand.relation_type)
            if key in by_key:
                # Already handled in boost pass.
                continue

            sources = ["vlm"]
            if spatial is not None and _contact_ok(spatial, cand.relation_type):
                sources.append("spatial")
            pose = pose_by_index.get(sub_i)
            if pose is not None and pose.action.replace(" ", "_") == cand.relation_type:
                sources.append("pose")

            # Require spatial contact corroboration — VLM alone never verifies.
            if "spatial" not in sources and "pose" not in sources:
                rejected_vlm += 1
                records.append(
                    InteractionEvidenceRecord(
                        sub_i,
                        obj_i,
                        cand.relation_type,
                        cand.confidence,
                        False,
                        tuple(sources),
                        "vlm_without_spatial_or_pose",
                        spatial,
                    )
                )
                continue

            # Mere proximity without contact still rejected (enforced by _contact_ok).
            conf = min(0.88, 0.62 + 0.10 * len(sources) + 0.08 * cand.confidence)
            by_key[key] = Relation(sub_i, obj_i, cand.relation_type, conf)
            recovered += 1
            records.append(
                InteractionEvidenceRecord(
                    sub_i,
                    obj_i,
                    cand.relation_type,
                    conf,
                    True,
                    tuple(sources),
                    "recovered_vlm_plus_corroboration",
                    spatial,
                )
            )

        # --- Pass 3: map eating/drinking VLM verbs onto holding cup/bottle when contact ---
        for cand in candidates:
            if cand.relation_type not in {"eating", "drinking"}:
                continue
            # Prefer promote holding when drink/food object is contact-supported.
            for person_i in person_indices:
                for node in nodes:
                    if node.label.lower() not in (
                        _FOOD_DRINK if cand.relation_type == "eating" else {"cup", "bottle", "wine glass", "bowl"}
                    ):
                        continue
                    if cand.object_phrase and not _label_match(cand.object_phrase, node.label):
                        continue
                    spatial = self._spatial_for_indices(
                        person_i, node.index, nodes, det_by_index, diagonal
                    )
                    if spatial is None or not _contact_ok(spatial, "holding"):
                        continue
                    hold_key = (person_i, node.index, "holding")
                    if hold_key not in by_key:
                        by_key[hold_key] = Relation(person_i, node.index, "holding", 0.72)
                        recovered += 1
                        records.append(
                            InteractionEvidenceRecord(
                                person_i,
                                node.index,
                                "holding",
                                0.72,
                                True,
                                ("vlm", "spatial"),
                                f"promoted_from_{cand.relation_type}",
                                spatial,
                            )
                        )

        fused = tuple(by_key.values())
        logger.info(
            "Interaction fusion: boosted=%d recovered=%d rejected_vlm=%d total_rels=%d",
            boosted,
            recovered,
            rejected_vlm,
            len(fused),
        )
        return InteractionFusionResult(
            relations=fused,
            records=tuple(records),
            boosted=boosted,
            recovered=recovered,
            rejected_vlm=rejected_vlm,
        )

    def apply_to_context(
        self,
        scene_context: SceneContext,
        fusion: InteractionFusionResult,
    ) -> SceneContext:
        """Return SceneContext with fused relations; keep other fields."""
        new_graph = SceneGraph(nodes=scene_context.graph.nodes, relations=fusion.relations)
        return SceneContext(
            graph=new_graph,
            attributes=scene_context.attributes,
            activities=scene_context.activities,
            environment=scene_context.environment,
            object_count=scene_context.object_count,
            dominant_objects=scene_context.dominant_objects,
            spatial_summary=scene_context.spatial_summary,
        )

    def _label_at(self, nodes: list[SceneNode], index: int) -> str:
        for node in nodes:
            if node.index == index:
                return node.label.lower()
        return ""

    def _spatial_for_indices(
        self,
        subject_index: int,
        object_index: int,
        nodes: list[SceneNode],
        det_by_index: dict[int, Detection],
        diagonal: float,
    ) -> SpatialSupport | None:
        person = det_by_index.get(subject_index)
        obj = det_by_index.get(object_index)
        if person is None or obj is None:
            return None
        if person.label.lower() not in _PERSON:
            # Swap if needed.
            if obj.label.lower() in _PERSON:
                person, obj = obj, person
            else:
                return None
        return compute_spatial_support(person, obj, diagonal=diagonal)

    def _ground_candidate(
        self,
        cand: VlmRelationCandidate,
        person_indices: list[int],
        nodes: list[SceneNode],
        det_by_index: dict[int, Detection],
        diagonal: float,
    ) -> tuple[int, int, SpatialSupport | None] | None:
        """Map VLM 'person holding phone' → (person_i, phone_i, spatial).

        Returns None when entity mapping is ambiguous or impossible.
        """
        if not person_indices:
            return None
        allowed: frozenset[str]
        for rel, _pat, labels in _INTERACTION_PATTERNS:
            if rel == cand.relation_type:
                allowed = labels
                break
        else:
            return None

        object_nodes = [
            n
            for n in nodes
            if n.label.lower() in allowed
            or (cand.object_phrase and _label_match(cand.object_phrase, n.label))
        ]
        if cand.object_phrase:
            phrase_matches = [
                n for n in object_nodes if _label_match(cand.object_phrase, n.label)
            ]
            if phrase_matches:
                object_nodes = phrase_matches
        if not object_nodes:
            return None

        # Score person–object pairs by spatial contact; pick unique best.
        scored: list[tuple[float, int, int, SpatialSupport | None]] = []
        for person_i in person_indices:
            for obj_node in object_nodes:
                if obj_node.index == person_i:
                    continue
                spatial = self._spatial_for_indices(
                    person_i, obj_node.index, nodes, det_by_index, diagonal
                )
                score = 0.0
                if spatial is not None:
                    score = (
                        spatial.proximity * 0.35
                        + spatial.iou * 0.35
                        + spatial.mask_overlap * 0.20
                        + (0.15 if spatial.hand_zone else 0.0)
                    )
                    if not _contact_ok(spatial, cand.relation_type if cand.relation_type not in {"eating", "drinking"} else "holding"):
                        # Still consider but heavily down-rank.
                        score *= 0.25
                scored.append((score, person_i, obj_node.index, spatial))

        if not scored:
            return None
        scored.sort(key=lambda row: row[0], reverse=True)
        best = scored[0]
        # Ambiguity: two close top scores with different entities → refuse.
        # Raise margin so overlapping people do not share one bag/vehicle claim.
        if len(scored) > 1 and best[0] > 0 and abs(best[0] - scored[1][0]) < 0.12:
            if best[1] != scored[1][1] or best[2] != scored[1][2]:
                return None
        if best[0] < 0.08 and best[3] is not None:
            # Extremely weak spatial — still return for rejection path upstream.
            return best[1], best[2], best[3]
        if best[3] is None and len(object_nodes) == 1 and len(person_indices) == 1:
            # No detections available — allow grounding by unique labels only;
            # caller must still refuse without spatial.
            return person_indices[0], object_nodes[0].index, None
        if best[0] < 0.08:
            return None
        return best[1], best[2], best[3]
