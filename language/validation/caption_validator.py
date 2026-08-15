"""Evidence validation for generated captions."""

import re

from core.contracts.analysis import SceneContext
from core.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Closed object lexicon for hallucination checks — not an English stopword allowlist.
# Ordinary verbs/adjectives outside this set cannot create false unsupported-object flags.
_OBJECT_CLAIM_LEXICON = frozenset(
    {
        "person",
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "crowd",
        "dog",
        "cat",
        "bird",
        "horse",
        "cow",
        "sheep",
        "bear",
        "zebra",
        "giraffe",
        "elephant",
        "bicycle",
        "bike",
        "motorcycle",
        "car",
        "bus",
        "truck",
        "train",
        "boat",
        "airplane",
        "plane",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "plate",
        "mug",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "sofa",
        "bed",
        "dining table",
        "table",
        "toilet",
        "tv",
        "television",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "fridge",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        # Concrete props/furniture (not broad place words like park/street/kitchen —
        # those are scene labels and must not create false unsupported-object flags).
        "cabinet",
        "counter",
        "shelf",
        "rack",
        "drawer",
        "lamp",
        "mirror",
        "curtain",
        "bag",
        "box",
        "basket",
        "tray",
        "pan",
        "pot",
        "stove",
        "dishwasher",
        "food",
        "meal",
        "drink",
        "coffee",
        "tea",
        "ball",
        "racket",
        "bat",
        "glove",
        "helmet",
        "shoe",
        "hat",
        "jacket",
        "shirt",
        "dress",
        "camera",
        "microphone",
        "speaker",
        "monitor",
        "screen",
        "tablet",
        "fence",
        "gate",
        "sign",
        "poster",
        "banner",
        "flag",
        "tree",
        "flower",
        "plant",
        "animal",
        "vehicle",
        "furniture",
        "appliance",
        # High-confidence hallucination nouns (rarely evidenced, often invented).
        "dragon",
        "unicorn",
        "spaceship",
        "dinosaur",
        "submarine",
        "helicopter",
        "rocket",
        "missile",
        "alien",
        "mermaid",
        "zombie",
    }
)


class CaptionEvidenceValidator:
    """Validates captions against scene graph evidence."""

    def allowed_vocabulary(self, context: SceneContext) -> set[str]:
        """Return lowercase tokens supported by structured scene evidence."""
        tokens: set[str] = set()
        for node in context.graph.nodes:
            tokens.update(part.lower() for part in node.label.split())
        for activity in context.activities.activities:
            tokens.update(part.lower() for part in activity.activity.split())
        for relation in context.graph.relations:
            tokens.update(relation.relation_type.replace("_", " ").split())
        # Attribute evidence (colors, clothing) must count as supported — not hallucinations.
        for attribute in context.attributes.attributes:
            tokens.update(re.findall(r"[a-zA-Z]{3,}", attribute.name.lower()))
            tokens.update(re.findall(r"[a-zA-Z]{3,}", attribute.value.lower()))
        environment = context.environment
        for value in (
            environment.scene_type,
            environment.setting,
            environment.indoor_outdoor,
            environment.social_context,
            environment.crowd_level,
            environment.weather,
            environment.time_of_day,
        ):
            tokens.update(part.lower() for part in value.split() if part)
        generic = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "with",
            "and",
            "or",
            "of",
            "to",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "there",
            "scene",
            "image",
            "content",
            "appears",
            "shows",
            "likely",
            "possibly",
            "maybe",
            "unknown",
            "uncertain",
            "people",
            "person",
            "present",
            "setting",
            "supported",
            "activity",
            "spatial",
            "relations",
            "identified",
            "level",
            "crowd",
            "gathering",
            "interaction",
            "complexity",
            "individual",
            "presence",
            "outdoor",
            "indoor",
            "several",
            "multiple",
            "group",
            "together",
            "visible",
            "positioned",
            "suggesting",
            "suggests",
            "appear",
            "standing",
            "seated",
            "leaning",
            "beside",
            "nearby",
            "across",
            "frame",
            "photograph",
            "captures",
            "depicts",
            "field",
            "court",
            "kitchen",
            "office",
            "street",
            "park",
            "room",
            "area",
            "space",
            "atmosphere",
            "weather",
            "daylight",
            "nighttime",
            "quiet",
            "lively",
            "distributed",
            "elements",
            "engaged",
            "active",
            "main",
            "among",
            "throughout",
            "share",
            "proximity",
            "relevant",
            "based",
            "detected",
            "reads",
            "resembles",
            "relationships",
            "spread",
            "creating",
            "within",
            "conversational",
            "distance",
            "surrounding",
            "human",
            "without",
            "distinct",
            "primary",
            "notable",
            "upper",
            "lower",
            "center",
            "left",
            "right",
            "side",
            "top",
            "bottom",
            "toward",
            "close",
            "farther",
            "overlap",
            "overlaps",
            "overlapping",
            "conditions",
            "lighting",
            "cues",
            "clear",
            "rainy",
            "snowy",
            "transportation",
            "vehicles",
            "vehicle",
            "corridor",
            "workspace",
            "reading",
            "dining",
            "tennis",
            "baseball",
            "soccer",
            "skateboarding",
            "surfing",
            "cycling",
            "skate",
            "ball",
            "frisbee",
            "flying",
            "kite",
            "skiing",
            "snowboarding",
            "preparing",
            "food",
            "conversation",
            "walking",
            "waiting",
            "driving",
            "working",
            "shopping",
            "photographed",
        }
        return tokens | generic

    def filter_unsupported_sentences(self, caption: str, context: SceneContext) -> str:
        """Remove sentences that introduce unsupported objects or activities."""
        allowed = self.allowed_vocabulary(context)
        supported_activities = {item.activity.lower() for item in context.activities.activities}
        kept: list[str] = []
        for sentence in self._split_sentences(caption):
            if self._sentence_supported(sentence, allowed, supported_activities, context):
                kept.append(sentence)
            else:
                logger.debug("Removed unsupported sentence: %s", sentence)
        if kept:
            return " ".join(kept)
        return self._fallback_uncertain(context)

    def unsupported_object_tokens(self, caption: str, context: SceneContext) -> tuple[str, ...]:
        """Return object-like claims in caption that are absent from evidence.

        Uses a closed object lexicon (not a huge English allowlist) so ordinary
        verbs/adjectives like preparing/food/compact cannot create false
        hallucination flags, while unsupported concrete nouns still flag.
        """
        allowed_objects = {node.label.lower() for node in context.graph.nodes}
        allowed_phrases = set(allowed_objects)
        for node in context.graph.nodes:
            allowed_phrases.update(part.lower() for part in node.label.split())
        for activity in context.activities.activities:
            allowed_phrases.add(activity.activity.lower().replace("_", " "))
            for part in activity.activity.lower().replace("_", " ").split():
                allowed_phrases.add(part)
        for attr in context.attributes.attributes:
            allowed_phrases.add(attr.value.lower())
            for part in attr.value.lower().split():
                allowed_phrases.add(part)
        for relation in context.graph.relations:
            allowed_phrases.add(relation.relation_type.lower().replace("_", " "))
        env = context.environment
        for value in (
            env.scene_type,
            env.setting,
            env.indoor_outdoor,
            env.social_context,
            env.crowd_level,
            env.weather,
            env.time_of_day,
        ):
            allowed_phrases.add(value.lower())
            for part in value.lower().split():
                allowed_phrases.add(part)

        caption_l = caption.lower()
        found: list[str] = []
        for claim in sorted(_OBJECT_CLAIM_LEXICON, key=len, reverse=True):
            if not re.search(rf"\b{re.escape(claim)}\b", caption_l):
                continue
            if any(claim == label or claim in label or label in claim for label in allowed_objects):
                continue
            if any(
                claim == phrase or claim in phrase or phrase in claim
                for phrase in allowed_phrases
                if len(phrase) >= 3
            ):
                continue
            # Soft synonym bridge for common scene nouns already evidenced.
            if claim in {"man", "woman", "boy", "girl", "child", "people", "person"} and any(
                x in allowed_objects for x in {"person", "people", "man", "woman", "boy", "girl", "child"}
            ):
                continue
            # Kitchen/dining context supports generic food/meal wording without inventing objects.
            if claim in {"food", "meal", "drink"} and any(
                x in allowed_phrases
                for x in {
                    "kitchen",
                    "bowl",
                    "plate",
                    "cup",
                    "oven",
                    "sink",
                    "stove",
                    "dining",
                    "refrigerator",
                    "fridge",
                    "preparing",
                }
            ):
                continue
            found.append(claim)
        return tuple(sorted(set(found)))

    def _sentence_supported(
        self,
        sentence: str,
        allowed: set[str],
        supported_activities: set[str],
        context: SceneContext,
    ) -> bool:
        lower = sentence.lower()
        unsupported = self.unsupported_object_tokens(sentence, context)
        if unsupported:
            return False
        if any(activity in lower for activity in supported_activities if activity):
            return True
        object_hits = [node.label.lower() for node in context.graph.nodes if node.label.lower() in lower]
        if object_hits:
            return True
        if any(marker in lower for marker in ("unknown", "uncertain", "possibly", "likely")):
            return True
        # Empty-evidence / scene-label sentences may keep mild extras; otherwise reject
        # sentences that introduce multiple tokens outside evidence vocabulary.
        tokens = re.findall(r"[a-z]{4,}", lower)
        extra = [token for token in tokens if token not in allowed]
        if context.object_count == 0:
            return len(extra) <= 2
        return len(extra) <= 1

    def _split_sentences(self, caption: str) -> tuple[str, ...]:
        cleaned = caption.strip()
        if not cleaned:
            return ()
        return tuple(part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip())

    def _fallback_uncertain(self, context: SceneContext) -> str:
        from language.prompts.context_caption import build_context_caption

        return build_context_caption(context).text
