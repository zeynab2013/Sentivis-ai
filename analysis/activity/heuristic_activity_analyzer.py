"""Infer evidence-backed activities from scene graph structure."""

from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import ActivityEvidence, ActivityHints, SceneGraph
from core.logging import get_logger

logger = get_logger(__name__)

_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_SPORT_ITEM_ACTIVITY: dict[str, str] = {
    # Literal interaction phrases — venue sports ("playing tennis") require stronger evidence.
    "tennis racket": "playing with a tennis racket",
    "baseball bat": "playing with a baseball bat",
    "sports ball": "playing with a ball",
    "skateboard": "skateboarding",
    "surfboard": "surfing",
    "kite": "flying a kite",
    "frisbee": "playing frisbee",
    "skis": "skiing",
    "snowboard": "snowboarding",
}
_SPORT_ITEMS = set(_SPORT_ITEM_ACTIVITY)
_VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
_FOOD_LABELS = {"dining table", "pizza", "cake", "bowl", "sandwich"}
_KITCHEN_LABELS = {"oven", "refrigerator", "sink", "microwave", "toaster", "bowl", "cup", "bottle"}
_TECH_LABELS = {"laptop", "cell phone", "tv", "keyboard", "mouse"}
_READ_LABELS = {"book", "newspaper"}


class HeuristicActivityAnalyzer:
    """Legacy heuristic activity inference (benchmark / comparison only)."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config

    def analyze(self, graph: SceneGraph) -> ActivityHints:
        if not graph.nodes:
            return ActivityHints(activities=(), confidence=self._config.activity.confidence_empty)

        labels_by_index = {node.index: node.label.lower() for node in graph.nodes}
        label_set = set(labels_by_index.values())
        person_indices = [index for index, label in labels_by_index.items() if label in _PERSON_LABELS]
        evidence_items: list[ActivityEvidence] = []

        evidence_items.extend(self._sport_activities(graph, person_indices, labels_by_index))
        evidence_items.extend(self._ball_sports(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._dining_activities(graph, person_indices, label_set))
        evidence_items.extend(self._drinking_activities(graph, person_indices, label_set))
        evidence_items.extend(self._cooking_activities(graph, person_indices, label_set))
        evidence_items.extend(self._technology_activities(graph, person_indices, label_set))
        evidence_items.extend(self._typing_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._reading_activities(graph, person_indices, labels_by_index))
        evidence_items.extend(self._classroom_activities(graph, person_indices, label_set))
        evidence_items.extend(self._shopping_activities(graph, person_indices, label_set))
        evidence_items.extend(self._pet_activities(graph, person_indices, labels_by_index))
        evidence_items.extend(self._horse_handling_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._feeding_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._campfire_activities(graph, person_indices, label_set))
        evidence_items.extend(self._repair_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._studying_activities(graph, person_indices, label_set))
        evidence_items.extend(self._driving_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._crossing_street_activities(graph, person_indices, label_set))
        evidence_items.extend(self._children_playing(graph, person_indices, labels_by_index))
        evidence_items.extend(self._transportation_activities(graph, label_set))
        evidence_items.extend(self._waiting_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._cycling_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._walking_activities(graph, person_indices, labels_by_index))
        evidence_items.extend(self._running_activities(graph, person_indices, label_set))
        evidence_items.extend(self._phone_activities(graph, person_indices, labels_by_index, label_set))
        evidence_items.extend(self._photo_activities(graph, person_indices, labels_by_index))
        evidence_items.extend(self._meeting_activities(graph, person_indices, label_set))
        evidence_items.extend(self._teaching_activities(graph, person_indices, label_set))
        evidence_items.extend(self._cleaning_activities(graph, person_indices, label_set))
        evidence_items.extend(self._exercise_activities(graph, person_indices, label_set))

        # Prefer an empty activity list over placeholder "people present" / "static scene".
        if not evidence_items:
            logger.debug("No evidence-backed activities inferred")
            return ActivityHints(activities=(), confidence=0.0)

        # Keep highest-confidence unique activity phrases (avoid typing+tech duplicates).
        best_by_name: dict[str, ActivityEvidence] = {}
        for item in evidence_items:
            key = item.activity.lower().strip()
            previous = best_by_name.get(key)
            if previous is None or item.confidence > previous.confidence:
                best_by_name[key] = item
        unique = list(best_by_name.values())
        overall = max(item.confidence for item in unique)
        logger.debug("Inferred %d activities with evidence", len(unique))
        return ActivityHints(activities=tuple(unique), confidence=overall)

    def _sport_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        results: list[ActivityEvidence] = []
        for relation in graph.relations:
            if relation.relation_type != "playing_with":
                continue
            subject_label = labels_by_index.get(relation.subject_index, "")
            object_label = labels_by_index.get(relation.object_index, "")
            if subject_label in _PERSON_LABELS and object_label in _SPORT_ITEMS:
                activity = _SPORT_ITEM_ACTIVITY.get(object_label, "playing sports")
                results.append(
                    ActivityEvidence(
                        activity=activity,
                        confidence=min(0.9, relation.confidence + 0.1),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=(
                            f"Person interacting with {object_label} "
                            f"via {relation.relation_type}."
                        ),
                    )
                )
        # Do not invent sport activities from bare co-occurrence of person + equipment.
        # Possession alone is rejected downstream; skip emitting empty-support guesses here.
        return results

    def _ball_sports(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or "sports ball" not in label_set:
            return []
        ball_nodes = [node.index for node in graph.nodes if node.label.lower() == "sports ball"]
        play_relations = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"playing_with", "using", "holding"}
            and (
                (relation.subject_index in person_indices and relation.object_index in ball_nodes)
                or (relation.object_index in person_indices and relation.subject_index in ball_nodes)
            )
        ]
        # Without a person–ball interaction, do not invent a sport activity.
        if not play_relations:
            return []
        # Prefer playing_with/using for ball play; holding alone → holding.
        strong = [r for r in play_relations if r.relation_type in {"playing_with", "using"}]
        if strong:
            # Keep literal ball interaction — do not rename to a specific sport from
            # co-present equipment (bat/racket ≠ baseball/tennis claim).
            activity = "playing with a ball"
            conf = 0.74 if len(person_indices) >= 2 else 0.70
            support = strong
        else:
            activity = "holding a sports ball"
            conf = 0.70
            support = play_relations
        return [
            ActivityEvidence(
                activity=activity,
                confidence=conf,
                supporting_node_indices=tuple(person_indices[:2] + ball_nodes[:1]),
                supporting_relation_types=tuple(rel.relation_type for rel in support[:2]),
                rationale="People interact with a sports ball in a recreational context.",
            )
        ]

    def _cooking_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or not (label_set & _KITCHEN_LABELS):
            return []
        kitchen_nodes = [node.index for node in graph.nodes if node.label.lower() in _KITCHEN_LABELS]
        # Require cookware interaction — holding a cup/bowl is not cooking.
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"using"}
            and (
                (
                    relation.subject_index in person_indices
                    and relation.object_index in kitchen_nodes
                )
                or (
                    relation.object_index in person_indices
                    and relation.subject_index in kitchen_nodes
                )
            )
        ]
        cookware = {"oven", "microwave", "refrigerator", "sink", "knife", "spoon", "fork"}
        cookware_nodes = [
            node.index for node in graph.nodes if node.label.lower() in cookware
        ]
        if not interactive:
            return []
        if not any(
            (
                rel.object_index in cookware_nodes
                or rel.subject_index in cookware_nodes
            )
            for rel in interactive
        ):
            return []
        return [
            ActivityEvidence(
                activity="cooking",
                confidence=0.80,
                supporting_node_indices=tuple(person_indices[:1] + kitchen_nodes[:2]),
                supporting_relation_types=tuple(rel.relation_type for rel in interactive[:2]),
                rationale="Person using kitchen appliances or utensils.",
            )
        ]

    def _typing_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        work_devices = {"keyboard", "laptop", "mouse"}
        if not person_indices or not (label_set & work_devices):
            return []
        device_nodes = [node.index for node in graph.nodes if node.label.lower() in work_devices]
        computer_context = bool(label_set & {"laptop", "tv", "mouse", "monitor", "keyboard"})
        # Prefer factual device use over occupational "working" claims.
        rich = (
            "using a laptop"
            if "laptop" in label_set
            else ("using a computer" if computer_context else "typing")
        )
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if (
                subject in _PERSON_LABELS
                and obj in work_devices
                and relation.relation_type == "using"
                and relation.confidence >= 0.62
            ):
                return [
                    ActivityEvidence(
                        activity=rich,
                        confidence=min(0.88, relation.confidence + 0.14),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale="Person using computing equipment.",
                    )
                ]
        # Co-occurrence of laptop + chair is NOT sufficient for "working".
        return []

    def _shopping_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Handbags / backpacks alone are possession, not shopping intent.
        # Require a shopping cart (direct retail signal) — never bags alone.
        if not person_indices or "shopping cart" not in label_set:
            return []
        cart_nodes = [
            node.index for node in graph.nodes if node.label.lower() == "shopping cart"
        ]
        cart_relations = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"pushing", "holding", "using", "near"}
            and (
                (relation.subject_index in person_indices and relation.object_index in cart_nodes)
                or (relation.object_index in person_indices and relation.subject_index in cart_nodes)
            )
        ]
        if not cart_relations and not cart_nodes:
            return []
        return [
            ActivityEvidence(
                activity="shopping",
                confidence=0.80 if cart_relations else 0.68,
                supporting_node_indices=tuple(person_indices[:1] + cart_nodes[:1]),
                supporting_relation_types=tuple(
                    rel.relation_type for rel in cart_relations[:2]
                )
                or ("near",),
                rationale="Person with a shopping cart.",
            )
        ]

    def _pet_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        pet_labels = {"dog", "cat", "bird"}
        results: list[ActivityEvidence] = []
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj in pet_labels:
                if relation.relation_type in {"feeding", "giving"}:
                    activity = f"feeding a {obj}" if obj != "bird" else "feeding birds"
                elif obj == "dog" and relation.relation_type in {
                    "leading",
                    "guiding",
                }:
                    activity = "walking a dog"
                elif obj == "dog" and relation.relation_type in {
                    "near",
                    "next_to",
                    "standing_beside",
                    "holding",
                    "carrying",
                }:
                    # Proximity / possession is not walking — leave as weak pet presence.
                    continue
                else:
                    # Generic "pet interaction" from weak relations is not evidence-safe.
                    continue
                results.append(
                    ActivityEvidence(
                        activity=activity,
                        confidence=min(0.82, relation.confidence + 0.12),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person interacting with {obj}.",
                    )
                )
        return results

    def _horse_handling_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or "horse" not in label_set:
            return []
        horse_nodes = [node.index for node in graph.nodes if node.label.lower() == "horse"]
        interactive = {
            "holding",
            "leading",
            "guiding",
            "carrying",
            "riding",
        }
        best: ActivityEvidence | None = None
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj == "horse" and relation.relation_type in interactive:
                if relation.relation_type == "riding":
                    activity = "riding a horse"
                    conf = min(0.90, relation.confidence + 0.18)
                elif relation.relation_type in {"leading", "guiding"}:
                    activity = "leading a horse"
                    conf = min(0.90, relation.confidence + 0.18)
                else:
                    # holding/carrying a horse without lead posture is not leading.
                    continue
                candidate = ActivityEvidence(
                    activity=activity,
                    confidence=conf,
                    supporting_node_indices=(relation.subject_index, relation.object_index),
                    supporting_relation_types=(relation.relation_type,),
                    rationale="Person–horse relationship in the scene graph.",
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        if best is not None:
            return [best]
        # Co-occurrence without an interaction relation is not an activity.
        return []

    def _studying_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices:
            return []
        study_set = {"book", "laptop", "keyboard", "mouse"}
        if "book" not in label_set:
            return []
        book_nodes = [node.index for node in graph.nodes if node.label.lower() == "book"]
        study_relations = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"looking_at", "holding", "using", "reading"}
            and (
                (relation.subject_index in person_indices and relation.object_index in book_nodes)
                or (relation.object_index in person_indices and relation.subject_index in book_nodes)
            )
        ]
        # Co-presence of a book with furniture is NOT studying — require book interaction.
        if not study_relations:
            return []
        nodes = [node.index for node in graph.nodes if node.label.lower() in study_set | {"chair", "dining table"}]
        return [
            ActivityEvidence(
                activity="reading a book" if all(
                    r.relation_type == "looking_at" for r in study_relations
                ) else "studying",
                confidence=0.80,
                supporting_node_indices=tuple(person_indices[:1] + nodes[:2]),
                supporting_relation_types=tuple(rel.relation_type for rel in study_relations[:2]),
                rationale="Person interacting with a book.",
            )
        ]

    def _driving_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        drive_labels = {"car", "bus", "truck"}
        if not person_indices or not (label_set & drive_labels):
            return []
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj in drive_labels and relation.relation_type in {
                "inside",
                "driving",
            }:
                return [
                    ActivityEvidence(
                        activity="driving",
                        confidence=min(0.84, relation.confidence + 0.1),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person inside {obj} as vehicle operator.",
                    )
                ]
        return []

    def _crossing_street_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Disabled: person near traffic objects is not evidence of crossing a street.
        _ = graph, person_indices, label_set
        return []

    def _children_playing(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        child_indices = [index for index in person_indices if labels_by_index.get(index) == "child"]
        if not child_indices:
            return []
        play_items = {"sports ball", "kite", "frisbee", "teddy bear", "skateboard"}
        play_nodes = [node.index for node in graph.nodes if node.label.lower() in play_items]
        if not play_nodes:
            return []
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"playing_with", "holding", "using"}
            and relation.confidence >= 0.62
            and (
                (relation.subject_index in child_indices and relation.object_index in play_nodes)
                or (relation.object_index in child_indices and relation.subject_index in play_nodes)
            )
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="children playing",
                confidence=0.78,
                supporting_node_indices=tuple(child_indices[:2] + play_nodes[:1]),
                supporting_relation_types=tuple(rel.relation_type for rel in interactive[:2]),
                rationale="Child interacting with play-related objects.",
            )
        ]

    def _dining_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or not (label_set & _FOOD_LABELS):
            return []
        food_nodes = [node.index for node in graph.nodes if node.label.lower() in _FOOD_LABELS]
        kitchen_nodes = [node.index for node in graph.nodes if node.label.lower() in _KITCHEN_LABELS]
        # Eating requires an eating/using relation on food — holding alone is possession.
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"eating", "using"}
            and relation.confidence >= 0.62
            and (
                (
                    relation.subject_index in person_indices
                    and (
                        relation.object_index in food_nodes
                        or relation.object_index in kitchen_nodes
                    )
                )
                or (
                    relation.object_index in person_indices
                    and (
                        relation.subject_index in food_nodes
                        or relation.subject_index in kitchen_nodes
                    )
                )
            )
        ]
        if not interactive:
            return []
        if label_set & _KITCHEN_LABELS and kitchen_nodes:
            return [
                ActivityEvidence(
                    activity="kitchen preparation",
                    confidence=0.78,
                    supporting_node_indices=tuple(person_indices[:1] + kitchen_nodes[:1]),
                    supporting_relation_types=tuple(rel.relation_type for rel in interactive[:2]),
                    rationale="Person using kitchen objects supports preparation.",
                )
            ]
        activity = "eating"
        return [
            ActivityEvidence(
                activity=activity,
                confidence=0.76,
                supporting_node_indices=tuple(person_indices[:2] + food_nodes[:2]),
                supporting_relation_types=tuple(rel.relation_type for rel in interactive[:2]),
                rationale="Person using food/dining objects supports eating.",
            )
        ]

    def _drinking_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        drink_labels = {"cup", "bottle", "wine glass", "bowl"}
        if not person_indices or not (label_set & drink_labels):
            return []
        drink_nodes = [node.index for node in graph.nodes if node.label.lower() in drink_labels]
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type == "drinking"
            and relation.confidence >= 0.62
            and relation.subject_index in person_indices
            and relation.object_index in drink_nodes
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="drinking",
                confidence=min(0.86, interactive[0].confidence + 0.10),
                supporting_node_indices=(interactive[0].subject_index, interactive[0].object_index),
                supporting_relation_types=(interactive[0].relation_type,),
                rationale="Person drinking from a container.",
            )
        ]

    def _technology_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or not (label_set & _TECH_LABELS):
            return []
        # Computing devices only — TV/phone co-occurrence with chairs is not office work.
        work_devices = {"laptop", "keyboard", "mouse"}
        if not (label_set & work_devices):
            return []
        tech_nodes = [node.index for node in graph.nodes if node.label.lower() in work_devices]
        relation_types = tuple(
            relation.relation_type
            for relation in graph.relations
            if relation.relation_type in {"holding", "looking_at", "using"}
            and relation.subject_index in person_indices
            and relation.object_index in tech_nodes
        )
        # Require an interaction relation — presence alone is not verified work.
        if not relation_types:
            return []
        # looking_at / holding a device supports use, not occupational "working".
        if "using" in relation_types:
            activity = (
                "using a laptop"
                if "laptop" in label_set
                else ("using a computer" if label_set & {"keyboard", "mouse"} else "using a computer")
            )
        else:
            activity = (
                "using a laptop"
                if "laptop" in label_set
                else "using a computer"
            )
        return [
            ActivityEvidence(
                activity=activity,
                confidence=0.78 if "using" in relation_types else 0.72,
                supporting_node_indices=tuple(person_indices[:1] + tech_nodes[:1]),
                supporting_relation_types=relation_types,
                rationale="Person interacting with a computing device.",
            )
        ]

    def _reading_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        results: list[ActivityEvidence] = []
        for relation in graph.relations:
            if relation.relation_type != "looking_at":
                continue
            subject_label = labels_by_index.get(relation.subject_index, "")
            object_label = labels_by_index.get(relation.object_index, "")
            if subject_label in _PERSON_LABELS and object_label in _READ_LABELS:
                results.append(
                    ActivityEvidence(
                        activity="reading",
                        confidence=min(0.88, relation.confidence + 0.15),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person looking at {object_label}.",
                    )
                )
        return results

    def _classroom_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices:
            return []
        study_objects = {"book", "laptop", "keyboard", "backpack"}
        if not (label_set & study_objects):
            return []
        # "Classroom learning" requires backpack + study material interaction.
        if "backpack" not in label_set:
            return []
        study_nodes = [node.index for node in graph.nodes if node.label.lower() in study_objects]
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"looking_at", "holding", "using", "reading"}
            and relation.subject_index in person_indices
            and relation.object_index in study_nodes
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="using study materials",
                confidence=0.74,
                supporting_node_indices=tuple(person_indices[:1] + study_nodes[:2]),
                supporting_relation_types=tuple(r.relation_type for r in interactive[:2]),
                rationale="Person interacting with backpack and study materials.",
            )
        ]

    def _transportation_activities(
        self,
        graph: SceneGraph,
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Driving is for enclosed motor vehicles — bicycle/motorcycle riding is cycling.
        drive_vehicles = {"car", "bus", "truck", "van", "train"}
        vehicle_nodes = [
            node.index for node in graph.nodes if node.label.lower() in drive_vehicles
        ]
        if not vehicle_nodes:
            return []
        labels_by_index = {node.index: node.label.lower() for node in graph.nodes}
        strong = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"inside", "driving"}
            and relation.confidence >= 0.65
            and (
                labels_by_index.get(relation.object_index, "") in drive_vehicles
                or labels_by_index.get(relation.subject_index, "") in drive_vehicles
            )
        ]
        if not strong:
            return []
        return [
            ActivityEvidence(
                activity="driving",
                confidence=0.72,
                supporting_node_indices=tuple(vehicle_nodes[:2]),
                supporting_relation_types=tuple(relation.relation_type for relation in strong[:3]),
                rationale="Person–vehicle cabin interaction detected.",
            )
        ]

    def _waiting_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices:
            return []
        roadish = {"road", "sidewalk", "street", "crosswalk", "car", "bus", "truck", "bench"}
        if not (label_set & roadish):
            return []
        # Prefer waiting when person is beside infrastructure/vehicle without riding/driving.
        strong_motion = {
            relation
            for relation in graph.relations
            if relation.relation_type in {"riding", "driving", "inside", "running"}
            and relation.confidence >= 0.6
        }
        if strong_motion:
            return []
        near_road = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"near", "next_to", "standing_beside", "beside"}
            and (
                labels_by_index.get(relation.object_index, "") in roadish
                or labels_by_index.get(relation.subject_index, "") in roadish
            )
        ]
        if not near_road and not (label_set & {"bench", "road", "sidewalk"}):
            return []
        return [
            ActivityEvidence(
                activity="waiting beside a road",
                confidence=0.68 if near_road else 0.58,
                supporting_node_indices=tuple(person_indices[:1]),
                supporting_relation_types=tuple(relation.relation_type for relation in near_road[:3]),
                rationale="Person beside road/vehicle infrastructure without motion relations.",
            )
        ]

    def _feeding_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        animals = {"horse", "dog", "cat", "bird", "cow", "sheep"}
        if not person_indices or not (label_set & animals):
            return []
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj in animals and relation.relation_type in {
                "feeding",
                "giving",
            }:
                return [
                    ActivityEvidence(
                        activity=f"feeding animals" if obj in {"cow", "sheep"} else f"feeding a {obj}",
                        confidence=min(0.84, relation.confidence + 0.12),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person–{obj} feeding interaction.",
                    )
                ]
        return []

    def _campfire_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Disabled: outdoor prop clusters are not evidence of preparing a campfire.
        _ = graph, person_indices, label_set
        return []

    def _repair_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        repair_targets = {"bicycle", "motorcycle", "car", "truck", "bus"}
        if not person_indices or not (label_set & repair_targets):
            return []
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if (
                subject in _PERSON_LABELS
                and obj in repair_targets
                and relation.relation_type in {"using", "holding", "looking_at"}
            ):
                # Riding/driving take precedence elsewhere.
                if any(
                    rel.relation_type in {"riding", "driving", "inside"}
                    and rel.subject_index == relation.subject_index
                    for rel in graph.relations
                ):
                    return []
                return [
                    ActivityEvidence(
                        activity="repairing equipment",
                        confidence=min(0.78, relation.confidence + 0.1),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person attending to {obj} without riding/driving.",
                    )
                ]
        return []

    def _walking_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        if len(person_indices) < 1:
            return []
        outdoor_labels = {"road", "sidewalk", "street", "crosswalk"}
        if not any(labels_by_index.get(index) in _PERSON_LABELS for index in person_indices):
            return []
        outdoor_nodes = [node.index for node in graph.nodes if node.label.lower() in outdoor_labels]
        if not outdoor_nodes:
            return []
        # Only emit walking when no stronger object-driven activity is likely.
        strong_labels = {
            "horse", "dog", "bicycle", "sports ball", "tennis racket", "laptop",
            "keyboard", "shopping cart", "book", "car", "bus", "truck",
        }
        if any(node.label.lower() in strong_labels for node in graph.nodes):
            return []
        return [
            ActivityEvidence(
                activity="walking",
                confidence=0.48,
                supporting_node_indices=tuple(person_indices[:1] + outdoor_nodes[:1]),
                supporting_relation_types=tuple(
                    relation.relation_type
                    for relation in graph.relations
                    if relation.relation_type in {"near", "standing_beside"}
                ),
                rationale="Person detected near outdoor path infrastructure.",
            )
        ]

    def _conversation_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
    ) -> list[ActivityEvidence]:
        # Disabled: co-presence alone is not evidence of conversation.
        _ = graph, person_indices
        return []

    def _cycling_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str] | None = None,
    ) -> list[ActivityEvidence]:
        bike_nodes = [
            node.index
            for node in graph.nodes
            if node.label.lower() in {"bicycle", "motorcycle"}
        ]
        if not bike_nodes or not person_indices:
            return []
        labels = label_set or set(labels_by_index.values())
        touring = bool(labels & {"backpack", "suitcase", "handbag"})
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj in {"bicycle", "motorcycle"}:
                # Require posture/interaction that supports the claim — not near/holding alone.
                if relation.relation_type in {"pushing", "pulling", "carrying"}:
                    activity = f"pushing a {obj}"
                elif relation.relation_type in {"riding", "sitting_on", "using"}:
                    activity = f"riding a {obj}"
                else:
                    # holding/near/next_to alone does not confirm riding.
                    continue
                # Bags do not rewrite cycling into touring/shopping.
                _ = touring
                return [
                    ActivityEvidence(
                        activity=activity,
                        confidence=min(0.9, relation.confidence + 0.14),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale=f"Person interacting with a {obj}.",
                    )
                ]
        # Co-occurrence without riding/pushing posture is not a cycling activity.
        return []

    def _running_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Running requires pose/motion evidence — outdoor presence alone is insufficient.
        _ = graph, person_indices, label_set
        return []

    def _phone_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or "cell phone" not in label_set:
            return []
        phone_nodes = [node.index for node in graph.nodes if node.label.lower() == "cell phone"]
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if (
                subject in _PERSON_LABELS
                and obj == "cell phone"
                and relation.relation_type in {"holding", "using", "looking_at"}
            ):
                return [
                    ActivityEvidence(
                        activity="using phone",
                        confidence=min(0.8, relation.confidence + 0.1),
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale="Person interacting with cell phone.",
                    )
                ]
        # Co-occurrence without interaction is not phone use.
        return []

    def _photo_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        labels_by_index: dict[int, str],
    ) -> list[ActivityEvidence]:
        for relation in graph.relations:
            subject = labels_by_index.get(relation.subject_index, "")
            obj = labels_by_index.get(relation.object_index, "")
            if subject in _PERSON_LABELS and obj == "cell phone" and relation.relation_type == "holding":
                return [
                    ActivityEvidence(
                        activity="taking photo",
                        confidence=0.55,
                        supporting_node_indices=(relation.subject_index, relation.object_index),
                        supporting_relation_types=(relation.relation_type,),
                        rationale="Person holding phone in photo posture.",
                    )
                ]
        return []

    def _meeting_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if len(person_indices) < 2:
            return []
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"talking_to", "looking_at"}
            and relation.subject_index in person_indices
            and relation.object_index in person_indices
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="meeting",
                confidence=0.72,
                supporting_node_indices=tuple(person_indices[:4]),
                supporting_relation_types=tuple(r.relation_type for r in interactive[:2]),
                rationale="Multiple people interacting in a meeting-like exchange.",
            )
        ]

    def _teaching_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        # Teaching requires interaction evidence — multi-person + chair is insufficient.
        if len(person_indices) < 2:
            return []
        if not (label_set & {"book", "laptop"}):
            return []
        # Teaching needs explicit interpersonal exchange — looking_at alone is too weak.
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type == "talking_to"
            and (
                relation.subject_index in person_indices
                or relation.object_index in person_indices
            )
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="talking with others",
                confidence=0.70,
                supporting_node_indices=tuple(person_indices[:3]),
                supporting_relation_types=tuple(r.relation_type for r in interactive[:2]),
                rationale="People talking near study/tech objects.",
            )
        ]

    def _cleaning_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices or not (label_set & {"sink", "bowl", "bottle", "spoon"}):
            return []
        sink_nodes = [node.index for node in graph.nodes if node.label.lower() == "sink"]
        utensil_nodes = [
            node.index
            for node in graph.nodes
            if node.label.lower() in {"bowl", "bottle", "spoon", "cup", "fork", "knife"}
        ]
        interactive = [
            relation
            for relation in graph.relations
            if relation.relation_type in {"holding", "using", "washing"}
            and relation.subject_index in person_indices
            and relation.object_index in set(sink_nodes + utensil_nodes)
        ]
        if not interactive:
            return []
        return [
            ActivityEvidence(
                activity="washing dishes",
                confidence=min(0.82, interactive[0].confidence + 0.10),
                supporting_node_indices=tuple(person_indices[:1] + (sink_nodes[:1] or utensil_nodes[:1])),
                supporting_relation_types=tuple(r.relation_type for r in interactive[:2]),
                rationale="Person interacting with sink or dishwashing utensils.",
            )
        ]

    def _exercise_activities(
        self,
        graph: SceneGraph,
        person_indices: list[int],
        label_set: set[str],
    ) -> list[ActivityEvidence]:
        if not person_indices:
            return []
        if "sports ball" in label_set:
            return []
        if label_set & {"tennis racket", "skateboard", "surfboard", "skis", "snowboard"}:
            equip_nodes = [
                node.index
                for node in graph.nodes
                if node.label.lower()
                in {"tennis racket", "skateboard", "surfboard", "skis", "snowboard"}
            ]
            interactive = [
                relation
                for relation in graph.relations
                if relation.relation_type in {"holding", "using", "carrying", "riding"}
                and relation.subject_index in person_indices
                and relation.object_index in equip_nodes
            ]
            if not interactive:
                return []
            return [
                ActivityEvidence(
                    activity="training",
                    confidence=0.74,
                    supporting_node_indices=tuple(person_indices[:2] + equip_nodes[:1]),
                    supporting_relation_types=tuple(r.relation_type for r in interactive[:2]),
                    rationale="Person interacting with sports or exercise equipment.",
                )
            ]
        if len(person_indices) >= 1 and label_set & {"bench", "chair"}:
            return []
        return []
