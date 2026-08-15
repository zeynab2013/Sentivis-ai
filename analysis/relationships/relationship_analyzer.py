"""Infer spatial and semantic relationships between detected objects."""

from analysis.common.geometry import (
    euclidean_distance,
    image_diagonal,
    intersection_over_union,
    is_inside,
    is_near,
)
from analysis.common.mask_geometry import mask_contains, mask_overlap_ratio
from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import Relation
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.logging import get_logger

logger = get_logger(__name__)

_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_HOLDABLE = {"cup", "bottle", "book", "cell phone", "laptop", "wine glass", "fork", "knife", "spoon"}
_SITTABLE = {"chair", "couch", "sofa", "bench", "bed"}
_WEARABLE = {"backpack", "handbag", "tie", "suitcase", "umbrella", "hat"}
_CARRIABLE = {"suitcase", "handbag", "backpack", "umbrella", "sports ball"}
_FOOD_DRINK = {"cup", "bottle", "wine glass", "bowl", "fork", "spoon", "knife", "dining table"}
_READABLE = {"book", "laptop", "cell phone", "newspaper"}
_SPORT_ITEMS = {"sports ball", "tennis racket", "skateboard", "surfboard", "kite", "baseball bat"}
_VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
_LEADABLE_ANIMALS = {"horse", "dog", "sheep", "cow", "elephant"}
_RIDEABLE = {"horse", "bicycle", "motorcycle", "elephant"}
_WORK_SURFACE = {"dining table", "desk", "table", "laptop"}
_INTERACTION_TYPES = frozenset(
    {
        "holding",
        "wearing",
        "sitting_on",
        "playing_with",
        "leading",
        "riding",
        "carrying",
        "using",
        "guiding",
    }
)
_CONTAINER_LABELS = {
    "car",
    "bus",
    "truck",
    "train",
    "boat",
    "airplane",
    "refrigerator",
    "oven",
    "microwave",
    "toilet",
    "bathtub",
    "sink",
    "backpack",
    "handbag",
    "suitcase",
    "bowl",
    "cup",
    "vase",
    "couch",
    "bed",
    "building",
    "room",
    "house",
}
# Rideables / sports gear are never containers — a bicycle cannot "contain" a rider.
_NEVER_CONTAINERS = {
    "sports ball",
    "tennis racket",
    "skateboard",
    "surfboard",
    "kite",
    "frisbee",
    "baseball bat",
    "cell phone",
    "book",
    "bottle",
    "fork",
    "knife",
    "spoon",
    "bicycle",
    "motorcycle",
    "scooter",
    "person",
    "people",
    "man",
    "woman",
    "child",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
_EXCLUSIVE_PERSON_OBJECT = frozenset(
    {"holding", "wearing", "carrying", "riding", "sitting_on", "using"}
)


class RelationshipAnalyzer:
    """Analyzes spatial and semantic relationships between detected objects."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config

    def analyze(self, detections: DetectionResult) -> tuple[Relation, ...]:
        items = detections.detections
        relations: list[Relation] = []
        diagonal = image_diagonal(detections)
        seen: set[tuple[int, int, str]] = set()

        for i, subject in enumerate(items):
            for j, obj in enumerate(items):
                if i == j:
                    continue
                for relation in self._relations_between(i, j, subject, obj, diagonal):
                    key = (relation.subject_index, relation.object_index, relation.relation_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(relation)

        relations = self._prefer_interactions(relations)
        relations = self._exclusive_person_object_links(relations, items)
        logger.debug("Found %d relationships", len(relations))
        return tuple(relations)

    def _prefer_interactions(self, relations: list[Relation]) -> list[Relation]:
        """Drop bare near/next_to when a meaningful interaction already links the pair."""
        interactive_pairs = {
            (rel.subject_index, rel.object_index)
            for rel in relations
            if rel.relation_type in _INTERACTION_TYPES
        }
        if not interactive_pairs:
            return relations
        filtered: list[Relation] = []
        for rel in relations:
            if rel.relation_type in {"near", "next_to"} and (
                (rel.subject_index, rel.object_index) in interactive_pairs
                or (rel.object_index, rel.subject_index) in interactive_pairs
            ):
                continue
            filtered.append(rel)
        return filtered

    def _exclusive_person_object_links(
        self,
        relations: list[Relation],
        items: list[Detection] | tuple[Detection, ...],
    ) -> list[Relation]:
        """Bind each exclusive object to at most one person (best confidence).

        Prevents scene-level double assignment such as two people both holding
        the same handbag, or two people both riding the same bicycle.
        """
        by_object: dict[tuple[int, str], list[Relation]] = {}
        kept: list[Relation] = []
        for rel in relations:
            if rel.relation_type not in _EXCLUSIVE_PERSON_OBJECT:
                kept.append(rel)
                continue
            sub = items[rel.subject_index] if 0 <= rel.subject_index < len(items) else None
            if sub is None or sub.label.lower() not in _PERSON_LABELS:
                kept.append(rel)
                continue
            key = (rel.object_index, rel.relation_type)
            by_object.setdefault(key, []).append(rel)
        for group in by_object.values():
            group.sort(key=lambda r: r.confidence, reverse=True)
            kept.append(group[0])
            # Prefer one possession verb per object: wearing vs carrying vs holding.
        # Collapse competing possession verbs on the same object to the strongest.
        possession = {"holding", "wearing", "carrying"}
        best_possession: dict[int, Relation] = {}
        remainder: list[Relation] = []
        for rel in kept:
            if rel.relation_type not in possession:
                remainder.append(rel)
                continue
            sub = items[rel.subject_index] if 0 <= rel.subject_index < len(items) else None
            if sub is None or sub.label.lower() not in _PERSON_LABELS:
                remainder.append(rel)
                continue
            prev = best_possession.get(rel.object_index)
            if prev is None or rel.confidence > prev.confidence:
                best_possession[rel.object_index] = rel
        return remainder + list(best_possession.values())

    def _relations_between(
        self,
        subject_index: int,
        object_index: int,
        subject: Detection,
        obj: Detection,
        diagonal: float,
    ) -> list[Relation]:
        rel_cfg = self._config.relationships
        subject_box = subject.bounding_box
        object_box = obj.bounding_box
        distance = euclidean_distance(subject_box, object_box)
        relations: list[Relation] = []

        if distance < rel_cfg.overlap_distance:
            relations.append(
                Relation(subject_index, object_index, "overlapping", rel_cfg.overlap_confidence)
            )

        iou = intersection_over_union(subject_box, object_box)
        subject_label = subject.label.lower()
        object_label = obj.label.lower()
        image_area = float(max(1, subject_box.area + object_box.area))
        if self._mask_inside(subject, obj, image_area):
            relations.append(
                Relation(subject_index, object_index, "inside", min(rel_cfg.max_confidence, 0.82))
            )
        elif self._mask_inside(obj, subject, image_area):
            relations.append(
                Relation(object_index, subject_index, "inside", min(rel_cfg.max_confidence, 0.82))
            )
        elif is_inside(object_box, subject_box) and self._valid_containment(
            subject_label, object_label, object_box, subject_box
        ):
            relations.append(
                Relation(subject_index, object_index, "inside", min(rel_cfg.max_confidence, 0.7 + iou * 0.2))
            )
        elif is_inside(subject_box, object_box) and self._valid_containment(
            object_label, subject_label, subject_box, object_box
        ):
            relations.append(
                Relation(object_index, subject_index, "inside", min(rel_cfg.max_confidence, 0.7 + iou * 0.2))
            )

        if is_near(subject_box, object_box, diagonal, rel_cfg.near_distance_ratio):
            proximity = 1.0 - min(1.0, distance / max(diagonal * rel_cfg.near_distance_ratio, 1.0))
            # Require stronger proximity before emitting weak spatial links (cuts hallucinations).
            if proximity >= 0.45:
                confidence = min(rel_cfg.max_confidence, 0.45 + proximity * 0.4)
                relations.append(Relation(subject_index, object_index, "near", confidence))
            if proximity >= 0.65:
                relations.append(
                    Relation(subject_index, object_index, "next_to", min(rel_cfg.max_confidence, 0.55 + proximity * 0.35))
                )

        dx = object_box.center_x - subject_box.center_x
        dy = object_box.center_y - subject_box.center_y
        axis_dominance = abs(abs(dx) - abs(dy)) / max(abs(dx) + abs(dy), 1.0)
        proximity_for_axis = 1.0 - min(1.0, distance / max(diagonal * rel_cfg.near_distance_ratio, 1.0))
        # Only emit left/right/above/below when geometry is decisive and objects are near.
        if proximity_for_axis >= 0.4 and axis_dominance >= 0.25:
            if abs(dx) > abs(dy):
                relation_type = "right_of" if dx > 0 else "left_of"
            else:
                relation_type = "below" if dy > 0 else "above"
            confidence = min(rel_cfg.max_confidence, 0.4 + proximity_for_axis * 0.35)
            relations.append(Relation(subject_index, object_index, relation_type, confidence))

        # Semantic interactions must use image-relative proximity. The legacy
        # overlap_distance*2.5 gate (often ~2.5px) prevented leading/using/holding
        # from ever firing on real photographs.
        semantic_limit = max(
            diagonal * rel_cfg.near_distance_ratio * 1.75,
            float(rel_cfg.overlap_distance) * 2.5,
        )
        if distance < semantic_limit:
            relations.extend(
                self._semantic_relations(subject_index, object_index, subject, obj, distance, diagonal)
            )
        return relations

    def _semantic_relations(
        self,
        subject_index: int,
        object_index: int,
        subject: Detection,
        obj: Detection,
        distance: float,
        diagonal: float,
    ) -> list[Relation]:
        rel_cfg = self._config.relationships
        subject_label = subject.label.lower()
        object_label = obj.label.lower()
        relations: list[Relation] = []
        # Wider semantic radius than pure "near" so farm/office interactions score.
        semantic_radius = max(diagonal * rel_cfg.near_distance_ratio * 1.75, 1.0)
        proximity = 1.0 - min(1.0, distance / semantic_radius)

        if subject_label in _PERSON_LABELS and object_label in _HOLDABLE and proximity >= 0.62:
            # Holding requires real box overlap — proximity alone is insufficient.
            iou = intersection_over_union(subject.bounding_box, obj.bounding_box)
            overlap_ratio = 0.0
            if subject.segmentation is not None and obj.segmentation is not None:
                image_area = float(
                    max(1.0, subject.bounding_box.area + obj.bounding_box.area)
                )
                overlap_ratio = mask_overlap_ratio(
                    subject.segmentation, obj.segmentation, image_area
                )
            hand_zone = (
                obj.bounding_box.center_y
                >= subject.bounding_box.center_y - subject.bounding_box.height * 0.15
                and obj.bounding_box.center_y
                <= subject.bounding_box.y_max + subject.bounding_box.height * 0.05
                and abs(obj.bounding_box.center_x - subject.bounding_box.center_x)
                <= subject.bounding_box.width * 0.75
            )
            if (iou >= 0.08 or overlap_ratio >= 0.05) and hand_zone and proximity >= 0.62:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "holding",
                        min(rel_cfg.max_confidence, 0.62 + max(iou, proximity) * 0.28),
                    )
                )

        if subject_label in _PERSON_LABELS and object_label in _WEARABLE and proximity >= 0.62:
            iou = intersection_over_union(subject.bounding_box, obj.bounding_box)
            overlap_ratio = 0.0
            if subject.segmentation is not None and obj.segmentation is not None:
                image_area = float(
                    max(1.0, subject.bounding_box.area + obj.bounding_box.area)
                )
                overlap_ratio = mask_overlap_ratio(
                    subject.segmentation, obj.segmentation, image_area
                )
            # Bags/backpacks need real overlap — proximity alone invents wearing.
            torso_zone = (
                obj.bounding_box.center_y
                >= subject.bounding_box.y_min + subject.bounding_box.height * 0.15
                and obj.bounding_box.center_y
                <= subject.bounding_box.y_min + subject.bounding_box.height * 0.85
                and abs(obj.bounding_box.center_x - subject.bounding_box.center_x)
                <= subject.bounding_box.width * 0.85
            )
            contact = iou >= 0.08 or overlap_ratio >= 0.05
            if contact and torso_zone:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "wearing",
                        min(rel_cfg.max_confidence, 0.58 + max(iou, proximity) * 0.28),
                    )
                )

        if subject_label in _VEHICLE_LABELS and object_label in _VEHICLE_LABELS and proximity >= 0.2:
            relations.append(
                Relation(
                    subject_index,
                    object_index,
                    "parked_beside",
                    min(rel_cfg.max_confidence, 0.45 + proximity * 0.35),
                )
            )

        # Walking/running-toward is too speculative from static boxes — omit unless very close.
        if subject_label in _PERSON_LABELS and object_label in {"car", "bus", "truck"} and proximity >= 0.7:
            relations.append(
                Relation(
                    subject_index,
                    object_index,
                    "near",
                    min(rel_cfg.max_confidence, 0.5 + proximity * 0.25),
                )
            )

        if (
            subject_label in _PERSON_LABELS
            and object_label in _SITTABLE
            and subject.bounding_box.center_y <= obj.bounding_box.center_y + obj.bounding_box.height * 0.15
            and proximity >= 0.45
            and subject.bounding_box.height / max(subject.bounding_box.width, 1.0) <= 1.55
        ):
            relations.append(
                Relation(
                    subject_index,
                    object_index,
                    "sitting_on",
                    min(rel_cfg.max_confidence, 0.55 + proximity * 0.35),
                )
            )

        if subject_label in _PERSON_LABELS and object_label in _PERSON_LABELS and proximity >= 0.78:
            # Standing-beside is weak; keep only for very close co-located people and
            # emit at medium confidence so caption/QA gates can drop it.
            relations.append(
                Relation(
                    subject_index,
                    object_index,
                    "standing_beside",
                    min(rel_cfg.max_confidence, 0.48 + proximity * 0.22),
                )
            )

        # looking_at / talking_to are NOT emitted from bbox proximity alone.
        # Static boxes cannot support gaze or speech with competition-grade certainty.

        if subject_label in _PERSON_LABELS and object_label in _SPORT_ITEMS and proximity >= 0.55:
            iou = intersection_over_union(subject.bounding_box, obj.bounding_box)
            # Contact required — proximity alone invents sport interaction.
            # Stronger overlap → playing_with; lighter contact → holding (not a sport claim).
            if iou >= 0.05:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "playing_with",
                        min(rel_cfg.max_confidence, 0.55 + proximity * 0.32),
                    )
                )
            elif iou >= 0.02:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "holding",
                        min(rel_cfg.max_confidence, 0.58 + proximity * 0.28),
                    )
                )

        # Interaction-first animal handling (leading / riding) — not bare proximity.
        if subject_label in _PERSON_LABELS and object_label in _LEADABLE_ANIMALS and proximity >= 0.22:
            subject_box = subject.bounding_box
            object_box = obj.bounding_box
            person_above_animal = subject_box.center_y <= object_box.center_y + object_box.height * 0.2
            horizontally_adjacent = (
                subject_box.x_max >= object_box.x_min - diagonal * 0.03
                and subject_box.x_min <= object_box.x_max + diagonal * 0.03
            )
            mounted = (
                object_label in _RIDEABLE
                and subject_box.center_y < object_box.center_y
                and subject_box.center_y > object_box.y_min + object_box.height * 0.12
                and abs(subject_box.center_x - object_box.center_x) < object_box.width * 0.32
                and subject_box.area < object_box.area * 0.75
            )
            if mounted and object_label in {"horse", "elephant", "motorcycle", "bicycle"}:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "riding",
                        min(rel_cfg.max_confidence, 0.62 + proximity * 0.28),
                    )
                )
            elif person_above_animal and horizontally_adjacent:
                predicate = "leading" if object_label in {"horse", "dog", "sheep"} else "guiding"
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        predicate,
                        min(rel_cfg.max_confidence, 0.62 + max(proximity, 0.40) * 0.28),
                    )
                )

        if subject_label in _PERSON_LABELS and object_label in _RIDEABLE and proximity >= 0.5:
            subject_box = subject.bounding_box
            object_box = obj.bounding_box
            if (
                object_label in {"bicycle", "motorcycle"}
                and abs(subject_box.center_x - object_box.center_x) < object_box.width * 0.65
                and subject_box.center_y <= object_box.y_max
                and subject_box.y_max >= object_box.center_y
            ):
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "riding",
                        min(rel_cfg.max_confidence, 0.60 + proximity * 0.28),
                    )
                )

        if subject_label in _PERSON_LABELS and object_label in _CARRIABLE and proximity >= 0.62:
            # Distinct from wearing: object hangs beside/below the torso with contact.
            iou = intersection_over_union(subject.bounding_box, obj.bounding_box)
            overlap_ratio = 0.0
            if subject.segmentation is not None and obj.segmentation is not None:
                image_area = float(
                    max(1.0, subject.bounding_box.area + obj.bounding_box.area)
                )
                overlap_ratio = mask_overlap_ratio(
                    subject.segmentation, obj.segmentation, image_area
                )
            contact = iou >= 0.05 or overlap_ratio >= 0.04
            hangs_below = (
                object_label not in _WEARABLE
                or obj.bounding_box.center_y > subject.bounding_box.center_y
            )
            side_or_below = (
                hangs_below
                and abs(obj.bounding_box.center_x - subject.bounding_box.center_x)
                <= subject.bounding_box.width * 0.95
            )
            if contact and side_or_below:
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "carrying",
                        min(rel_cfg.max_confidence, 0.55 + max(iou, proximity) * 0.28),
                    )
                )

        if subject_label in _PERSON_LABELS and object_label in {"laptop", "keyboard", "mouse"} and proximity >= 0.58:
            iou = intersection_over_union(subject.bounding_box, obj.bounding_box)
            in_front = (
                abs(obj.bounding_box.center_x - subject.bounding_box.center_x)
                < max(subject.bounding_box.width, obj.bounding_box.width) * 0.85
                and obj.bounding_box.center_y >= subject.bounding_box.center_y - subject.bounding_box.height * 0.15
            )
            if in_front and (iou >= 0.02 or proximity >= 0.70):
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "using",
                        min(rel_cfg.max_confidence, 0.60 + proximity * 0.28),
                    )
                )

        if (
            subject_label in _PERSON_LABELS
            and object_label in _WORK_SURFACE
            and proximity >= 0.55
        ):
            # Desk/table in front of a person is spatial context, not "looking_at".
            # Prefer sitting_on / using (laptop) elsewhere; do not invent gaze.
            in_front = (
                abs(obj.bounding_box.center_x - subject.bounding_box.center_x)
                < max(subject.bounding_box.width, obj.bounding_box.width) * 0.9
                and obj.bounding_box.center_y >= subject.bounding_box.center_y - subject.bounding_box.height * 0.1
            )
            if in_front and object_label == "laptop":
                relations.append(
                    Relation(
                        subject_index,
                        object_index,
                        "using",
                        min(rel_cfg.max_confidence, 0.58 + proximity * 0.28),
                    )
                )

        # talking_to removed: face-to-face proximity ≠ conversation evidence.

        if (subject_label in _VEHICLE_LABELS or object_label in _VEHICLE_LABELS) and proximity >= 0.15:
            relations.append(
                Relation(
                    subject_index,
                    object_index,
                    "near_vehicle",
                    min(rel_cfg.max_confidence, 0.4 + proximity * 0.35),
                )
            )

        building_labels = {"building", "room", "house"}
        if subject_label in building_labels and not is_inside(subject.bounding_box, obj.bounding_box):
            relations.append(
                Relation(object_index, subject_index, "outside", min(rel_cfg.max_confidence, 0.65))
            )
        elif object_label in building_labels and not is_inside(obj.bounding_box, subject.bounding_box):
            relations.append(
                Relation(subject_index, object_index, "outside", min(rel_cfg.max_confidence, 0.65))
            )

        behind_front = self._depth_relation(subject, obj)
        if behind_front is not None:
            relation_type, confidence = behind_front
            relations.append(Relation(subject_index, object_index, relation_type, confidence))

        return relations

    def _depth_relation(self, subject: Detection, obj: Detection) -> tuple[str, float] | None:
        subject_area = subject.bounding_box.area
        object_area = obj.bounding_box.area
        if subject_area <= 0 or object_area <= 0:
            return None
        area_ratio = subject_area / object_area
        distance = euclidean_distance(subject.bounding_box, obj.bounding_box)
        if distance > self._config.relationships.overlap_distance * 4:
            return None
        if area_ratio < 0.65:
            return ("behind", min(self._config.relationships.max_confidence, 0.55))
        if area_ratio > 1.5:
            return ("in_front_of", min(self._config.relationships.max_confidence, 0.55))
        return None

    def _valid_containment(
        self,
        inner_label: str,
        outer_label: str,
        outer_box: BoundingBox,
        inner_box: BoundingBox,
    ) -> bool:
        """Return True when inner_label plausibly lies inside outer_label."""
        if inner_label in _PERSON_LABELS and outer_label in _PERSON_LABELS:
            return False
        if outer_label in _NEVER_CONTAINERS:
            return False
        if outer_box.area < inner_box.area * 1.15:
            return False
        if outer_label in _CONTAINER_LABELS:
            return True
        return outer_box.area >= inner_box.area * 2.0

    def _mask_inside(self, outer: Detection, inner: Detection, image_area: float) -> bool:
        """Segmentation-aware containment check."""
        if outer.segmentation is None or inner.segmentation is None:
            return False
        inner_label = inner.label.lower()
        outer_label = outer.label.lower()
        if inner_label in _PERSON_LABELS and outer_label in _NEVER_CONTAINERS:
            return False
        if inner_label in _PERSON_LABELS and outer_label in _PERSON_LABELS:
            return False
        if outer_label in _NEVER_CONTAINERS:
            return False
        overlap = mask_overlap_ratio(outer.segmentation, inner.segmentation, image_area)
        if overlap < 0.35:
            return False
        return mask_contains(outer.segmentation, inner.segmentation)
