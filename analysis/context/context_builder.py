"""Aggregate scene understanding into SceneContext from the scene graph."""

from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    SceneContext,
    SceneGraph,
)
from core.logging import get_logger

logger = get_logger(__name__)

_OUTDOOR_LABELS = {
    "tree",
    "sky",
    "mountain",
    "grass",
    "road",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "sidewalk",
    "street",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "dog",
    "horse",
    "bear",
    "cow",
    "sheep",
    "elephant",
    "zebra",
    "giraffe",
    "cat",
    "backpack",
    "umbrella",
    "kite",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "baseball bat",
    "skateboard",
    "surfboard",
    "tennis racket",
}
_INDOOR_LABELS = {
    "chair",
    "couch",
    "bed",
    "tv",
    "laptop",
    "dining table",
    "refrigerator",
    "microwave",
    "sink",
    "toilet",
    "oven",
    "toaster",
    "clock",
    "vase",
    "book",
    "keyboard",
    "mouse",
    "remote",
    "cell phone",
    "cup",
    "bowl",
    "spoon",
    "fork",
    "knife",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}
_SPORT_RECREATIONAL = {
    "sports ball",
    "tennis racket",
    "skateboard",
    "surfboard",
    "kite",
    "baseball bat",
    "frisbee",
    "skis",
    "snowboard",
}
_KITCHEN_LABELS = {"oven", "refrigerator", "sink", "microwave", "toaster", "bowl", "cup", "bottle"}
_CLASSROOM_LABELS = {"book", "chair", "backpack", "laptop", "keyboard"}
_HOSPITAL_LABELS = {"bed", "chair", "tv", "bottle", "cup", "clock"}
_WAREHOUSE_LABELS = {"truck", "forklift", "suitcase", "backpack"}
_AIRPORT_LABELS = {"airplane", "suitcase", "backpack", "handbag", "person"}
_PARK_LABELS = {"bench", "bird", "dog", "tree", "kite", "frisbee"}
_BEACH_LABELS = {"surfboard", "umbrella", "person"}
_FOOTBALL_FIELD = {"sports ball", "person"}
_BASKETBALL_COURT = {"sports ball", "person"}
_TENNIS_COURT = {"tennis racket", "person"}
_RESTAURANT_LABELS = {"dining table", "chair", "wine glass", "cup", "bowl", "fork", "knife"}
_CAFE_LABELS = {"cup", "chair", "dining table", "person"}
_OFFICE_LABELS = {"laptop", "keyboard", "mouse", "chair", "tv", "book"}
_MEETING_ROOM = {"chair", "laptop", "tv", "person"}
# Lab cues must be fixture-like — never treat person/book as laboratory evidence.
_LAB_FIXTURE_LABELS = {"bottle", "bowl", "cup"}
# Never include bare "person" — that mislabeled every outdoor human scene as playground.
_PLAYGROUND_CUES = {"sports ball", "kite", "skateboard", "frisbee"}
_FARM_LABELS = {"cow", "horse", "sheep", "goat"}
# Intersection cues must NOT include bare "person" — that matches every human scene.
_CROSSWALK = {"traffic light", "stop sign"}
_HIGHWAY = {"car", "truck", "bus", "motorcycle"}
_PARKING_LOT = {"parking meter"}
_CONSTRUCTION = {"stop sign"}
_SHOPPING = {"handbag", "suitcase", "dining table"}
_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_NIGHT_LABELS = {"moon", "street light"}
_DAY_LABELS = {"sun", "sky"}
_WEATHER_WET = {"umbrella", "rain"}
_WEATHER_SNOW = {"snowboard", "skis"}


class ContextBuilder:
    """Builds evidence-based scene context from the canonical scene graph."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config

    def build(
        self,
        graph: SceneGraph,
        attributes: AttributeSet,
        activities: ActivityHints,
    ) -> SceneContext:
        labels = [node.label for node in graph.nodes]
        dominant = tuple(sorted(set(labels), key=labels.count, reverse=True)[:5])
        environment = self._infer_environment(graph, activities)
        spatial_summary = self._spatial_summary(graph)

        context = SceneContext(
            graph=graph,
            attributes=attributes,
            activities=activities,
            environment=environment,
            object_count=len(graph.nodes),
            dominant_objects=dominant,
            spatial_summary=spatial_summary,
        )
        logger.info("Built scene context with %d objects", context.object_count)
        return context

    def _infer_environment(self, graph: SceneGraph, activities: ActivityHints) -> EnvironmentInfo:
        labels = {node.label.lower() for node in graph.nodes}
        evidence: list[str] = []
        activity_names = {item.activity.lower() for item in activities.activities}

        outdoor_score = len(labels & _OUTDOOR_LABELS)
        indoor_score = len(labels & _INDOOR_LABELS)
        if any(relation.relation_type == "near_vehicle" for relation in graph.relations):
            outdoor_score += 2
            evidence.append("Vehicle proximity detected in scene graph.")
        if "playing sports" in activity_names or "playing tennis" in activity_names:
            outdoor_score += 1
            evidence.append("Sports activity inferred from object interactions.")
        if activity_names & {"dining", "preparing food", "working"}:
            indoor_score += 2
            evidence.append(
                f"Indoor-oriented activity inferred: "
                f"{activity_names & {'dining', 'working', 'preparing food'}}."
            )
        if "transportation scene" in activity_names or "driving" in activity_names:
            outdoor_score += 2
            evidence.append("Transportation activity inferred.")
        if labels & _KITCHEN_LABELS:
            indoor_score += 3
            evidence.append("Kitchen appliance or utensil labels detected.")
        if labels & _CLASSROOM_LABELS and labels & _PERSON_LABELS:
            indoor_score += 2
            evidence.append("Indoor furniture/devices with people detected.")
        if labels & (_INDOOR_LABELS | _KITCHEN_LABELS | _CLASSROOM_LABELS) and labels & {"umbrella"}:
            indoor_score += 1
            outdoor_score = max(0, outdoor_score - 1)
            evidence.append("Indoor scene objects outweigh umbrella-only outdoor cue.")

        if outdoor_score > indoor_score:
            indoor_outdoor = "outdoor"
            scene_type, setting = self._specific_outdoor_setting(labels, activity_names)
            evidence.append("Outdoor object labels present.")
        elif indoor_score > 0:
            indoor_outdoor = "indoor"
            scene_type, setting = self._specific_indoor_setting(labels, activity_names)
            evidence.append("Indoor furniture or appliance labels present.")
        elif labels & _SPORT_RECREATIONAL:
            indoor_outdoor = "unknown"
            scene_type = "sports field"
            setting = "recreational area"
            evidence.append("Sports or recreational objects detected.")
        elif labels:
            indoor_outdoor = "unknown"
            scene_type, setting = self._specific_ambiguous_setting(labels)
            evidence.append("Objects detected without strong indoor/outdoor label signal.")
        else:
            indoor_outdoor = "unknown"
            # Omit placeholder settings — unknown is filtered from captions/UI.
            scene_type = "unknown"
            setting = "unknown"
            evidence.append("No strong indoor/outdoor label signal.")

        person_count = sum(1 for node in graph.nodes if node.label.lower() in _PERSON_LABELS)
        if person_count >= self._config.context.crowd_threshold:
            crowd_level = "crowded"
            social_context = "group gathering"
            evidence.append(f"{person_count} people detected.")
        elif person_count == 2:
            crowd_level = "pair"
            social_context = "small social interaction"
            evidence.append("Two people detected.")
        elif person_count == 1:
            crowd_level = "single person"
            social_context = "individual presence"
            evidence.append("One person detected.")
        else:
            crowd_level = "empty"
            social_context = "no people detected"
            evidence.append("No people detected.")

        relation_count = len(graph.relations)
        if relation_count >= self._config.context.complexity_high_relations:
            scene_complexity = "high"
        elif relation_count >= self._config.context.complexity_medium_relations:
            scene_complexity = "medium"
        else:
            scene_complexity = "low"
        evidence.append(f"{relation_count} relationships in scene graph.")

        if labels & _NIGHT_LABELS:
            time_of_day = "night"
            evidence.append("Night-related objects detected.")
        elif labels & _DAY_LABELS and outdoor_score > 0:
            time_of_day = "daytime"
            evidence.append("Daytime outdoor cues detected.")
        else:
            time_of_day = "unknown"

        if labels & _WEATHER_WET:
            weather = "rainy"
            evidence.append("Rain-related objects detected.")
        elif labels & _WEATHER_SNOW:
            weather = "snowy"
            evidence.append("Snow-related objects detected.")
        elif outdoor_score > 0:
            weather = "clear"
            evidence.append("Outdoor scene without rain/snow cues.")
        else:
            weather = "unknown"

        if activities.activities:
            evidence.append(
                "Activities: " + ", ".join(item.activity for item in activities.activities[:3])
            )

        atmosphere = self._infer_atmosphere(crowd_level, social_context, activity_names, scene_complexity)
        evidence.append(f"Atmosphere: {atmosphere}.")

        return EnvironmentInfo(
            scene_type=scene_type,
            setting=setting,
            time_of_day=time_of_day,
            weather=weather,
            indoor_outdoor=indoor_outdoor,
            social_context=social_context,
            crowd_level=crowd_level,
            scene_complexity=scene_complexity,
            evidence=tuple(evidence),
        )

    def _spatial_summary(self, graph: SceneGraph) -> str:
        if not graph.nodes:
            return "No objects detected in the scene."
        node_labels = ", ".join(f"{node.label} ({node.position_zone})" for node in graph.nodes[:6])
        semantic_relations = [
            relation
            for relation in graph.relations
            if relation.relation_type
            not in {"left_of", "right_of", "above", "below", "near", "far"}
        ]
        if semantic_relations:
            relation_text = "; ".join(
                f"{graph.nodes[relation.subject_index].label} "
                f"{relation.relation_type.replace('_', ' ')} "
                f"{graph.nodes[relation.object_index].label}"
                for relation in semantic_relations[:4]
                if relation.subject_index < len(graph.nodes)
                and relation.object_index < len(graph.nodes)
            )
            return f"Objects include {node_labels}. Key relations: {relation_text}."
        return f"Objects include {node_labels}. {len(graph.relations)} spatial relations identified."

    def _specific_outdoor_setting(self, labels: set[str], activity_names: set[str]) -> tuple[str, str]:
        # Named sport venues need more than bare equipment co-presence.
        sport_equipment = labels & _SPORT_RECREATIONAL
        if sport_equipment and labels & _PERSON_LABELS:
            verified_play = any(
                name.startswith("playing")
                or name in {"skiing", "snowboarding", "skateboarding", "surfing"}
                for name in activity_names
            )
            if verified_play and "tennis racket" in labels and any(
                name == "playing tennis" or name.startswith("playing tennis")
                for name in activity_names
            ):
                return "tennis court", "tennis court"
            if verified_play and "sports ball" in labels and len(labels & _PERSON_LABELS) >= 2:
                return "sports field", "outdoor sports field"
            if verified_play:
                return "outdoor scene", "recreational area"
            # Equipment without verified play → broad outdoor, not a named venue.
            return "outdoor scene", "recreational area"
        if labels & _BEACH_LABELS and "surfboard" in labels:
            return "beach", "beach"
        if labels & _BEACH_LABELS and "umbrella" in labels and "person" in labels:
            return "beach", "beach"
        # Farm/pasture before recreational heuristics — require multi-cue livestock evidence.
        farm_animals = labels & _FARM_LABELS
        if len(farm_animals) >= 2 or (farm_animals and labels & {"fence", "barn", "tractor"}):
            return "farm", "farm pasture"
        if farm_animals and labels & _PERSON_LABELS:
            # Single livestock with people → broad outdoor, not farm venue.
            return "outdoor scene", "outdoor area"
        wildlife = labels & {"bear", "elephant", "zebra", "giraffe", "bird"}
        if wildlife and not labels & {"car", "bus", "truck", "chair", "couch"}:
            return "natural environment", "outdoor area"
        if "mountain" in labels:
            return "mountain", "mountain landscape"
        if "tree" in labels and labels & {"bird", "bench"} and not labels & {"car", "bus", "truck"}:
            return "forest", "wooded area"
        if labels & _PLAYGROUND_CUES and labels & _PERSON_LABELS:
            return "playground", "playground"
        if labels & _CROSSWALK and labels & {"car", "bus", "truck", "person"}:
            return "crosswalk", "crosswalk"
        # Vehicles alone are not a highway — need road infrastructure cues.
        if labels & _HIGHWAY and labels & {"road", "traffic light", "stop sign"}:
            return "street", "city street"
        if labels & _HIGHWAY and not labels & _PERSON_LABELS:
            return "urban environment", "outdoor area"
        if "parking meter" in labels and labels & {"car", "truck", "bus"}:
            return "parking lot", "parking lot"
        if "airplane" in labels:
            return "airport", "airport"
        if "train" in labels:
            return "train station", "train station"
        if ("shopping cart" in labels or activity_names & {"shopping"}) and labels & _PERSON_LABELS:
            return "shopping mall", "shopping mall"
        if labels & _CONSTRUCTION and labels & {"truck", "person"}:
            return "construction site", "construction site"
        if labels & _PARK_LABELS:
            return "park", "park"
        if labels & {"car", "truck", "bus", "traffic light", "parking meter"}:
            if "parking meter" in labels:
                return "parking lot", "parking lot"
            return "street", "city street"
        if labels & {"bench", "bird", "dog"} and "tree" not in labels:
            return "park", "park"
        if "bicycle" in labels and labels & {"mountain", "tree", "grass", "sky"}:
            return "natural environment", "outdoor area"
        if "bicycle" in labels:
            return "outdoor scene", "outdoor area"
        if "surfboard" in labels:
            return "beach", "beach"
        if activity_names & {"transportation scene", "driving", "crossing street"}:
            return "street", "transportation corridor"
        return "outdoor scene", "outdoor area"

    def _specific_indoor_setting(self, labels: set[str], activity_names: set[str]) -> tuple[str, str]:
        kitchen_hits = labels & _KITCHEN_LABELS
        # Appliance evidence is stronger than dining-furniture venue guesses.
        if len(kitchen_hits) >= 2 or ("oven" in labels or "refrigerator" in labels or "sink" in labels):
            return "kitchen", "kitchen"
        # Restaurant needs distinctive serviceware — not merely table/chair/cup.
        restaurant_service = labels & {"wine glass", "fork", "knife", "spoon"}
        if "dining table" in labels and restaurant_service and not kitchen_hits:
            return "restaurant", "restaurant"
        cafe_service = labels & {"cup", "wine glass"}
        if cafe_service and "dining table" in labels and "laptop" not in labels and not kitchen_hits:
            if labels & {"fork", "knife", "spoon", "wine glass"}:
                return "cafe", "cafe"
        if labels & _HOSPITAL_LABELS and "bed" in labels:
            return "hospital", "hospital room"
        if labels & _WAREHOUSE_LABELS and "truck" in labels:
            return "warehouse", "warehouse"
        if "couch" in labels and ("tv" in labels or "remote" in labels) and not kitchen_hits:
            return "living room", "living room"
        if "bed" in labels and "chair" not in labels:
            return "bedroom", "bedroom"
        # Laboratory: require all three fixture-like cues (never person/book).
        if len(labels & _LAB_FIXTURE_LABELS) >= 3:
            return "laboratory", "laboratory"
        # Classroom needs backpack + study materials — laptop+book+chair alone is insufficient.
        if (
            "backpack" in labels
            and labels & _PERSON_LABELS
            and "book" in labels
            and labels & {"laptop", "keyboard", "chair"}
        ):
            return "classroom", "classroom"
        # Office requires multiple computing cues — a single laptop is not an office.
        if len(labels & {"laptop", "keyboard", "mouse"}) >= 2:
            return "office", "office workspace"
        if "laptop" in labels and labels & {"keyboard", "mouse", "monitor"}:
            return "office", "office workspace"
        # Meeting room after office cues — multi-person + shared display/device.
        if labels & _MEETING_ROOM and len(labels & _PERSON_LABELS) >= 2 and labels & {"laptop", "tv"}:
            return "meeting room", "meeting room"
        # Do NOT map book+chair+laptop → library/classroom — those need Tier-1 cues.
        if "dining table" in labels and not kitchen_hits:
            return "indoor scene", "indoor room"
        if "book" in labels and "chair" in labels:
            return "indoor scene", "indoor room"
        if activity_names & {"kitchen preparation", "cooking"}:
            return "kitchen", "kitchen"
        # Device interaction is not enough to force "office" as a venue.
        if activity_names & {
            "working at a computer",
            "typing",
            "working on a laptop",
            "using a laptop",
            "using a computer",
        }:
            if len(labels & {"laptop", "keyboard", "mouse"}) >= 2:
                return "office", "office"
            return "indoor scene", "indoor room"
        if activity_names & {"restaurant dining", "eating"} and restaurant_service:
            return "restaurant", "restaurant"
        return "indoor scene", "indoor room"

    def _specific_ambiguous_setting(self, labels: set[str]) -> tuple[str, str]:
        if labels & _SPORT_RECREATIONAL:
            return "outdoor scene", "recreational area"
        if labels & _KITCHEN_LABELS:
            return "kitchen", "kitchen"
        if labels & {"bear", "elephant", "zebra", "giraffe", "bird", "horse", "cow"}:
            return "natural environment", "outdoor area"
        if labels & {"car", "bus", "truck"}:
            return "urban environment", "outdoor area"
        return "unknown", "unknown"

    @staticmethod
    def _infer_atmosphere(
        crowd_level: str,
        social_context: str,
        activity_names: set[str],
        scene_complexity: str,
    ) -> str:
        if any(name.startswith("playing") for name in activity_names):
            return "competitive"
        if crowd_level in {"crowded", "busy"}:
            return "busy"
        if social_context in {"group gathering", "small social interaction"}:
            return "casual"
        if activity_names & {"working", "office work", "meeting", "teaching"}:
            return "formal"
        if scene_complexity == "low" and crowd_level == "empty":
            return "quiet"
        if activity_names & {"restaurant dining", "eating", "drinking"}:
            return "relaxed"
        return "neutral"
