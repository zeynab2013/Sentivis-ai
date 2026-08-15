"""Evidence-backed semantic enrichment for narrative synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.analysis import SceneContext

_PERSON = {"person", "people", "man", "woman", "child"}
_SPORT_BALL = {"sports ball"}
_FOOTBALL_CUE = {"sports ball"}  # COCO generic ball
_BASKETBALL_CUE: set[str] = set()  # no separate label in COCO
_TENNIS = {"tennis racket"}
_KITCHEN = {"oven", "refrigerator", "sink", "microwave", "toaster"}
_TECH = {"laptop", "keyboard", "mouse", "cell phone"}
_PETS = {"dog", "cat", "bird"}
_CHILD = {"child"}


@dataclass(frozen=True)
class SceneEnrichment:
    """Derived semantic hints — never override detections."""

    age_group: str
    crowd_behaviour: str
    emotional_atmosphere: str
    scene_purpose: str
    environment_type: str


def enrich_scene(context: SceneContext) -> SceneEnrichment:
    """Infer presentation hints only when evidence supports them."""
    labels = {node.label.lower() for node in context.graph.nodes}
    env = context.environment
    activities = {item.activity.lower() for item in context.activities.activities}
    person_count = sum(1 for node in context.graph.nodes if node.label.lower() in _PERSON)

    age_group = "unknown"
    if labels & _CHILD:
        age_group = "child"
    elif person_count == 1:
        age_group = "adult"
    elif person_count >= 2:
        age_group = "mixed group"

    crowd_behaviour = "unknown"
    if "having a conversation" in activities:
        crowd_behaviour = "conversation"
    elif "waiting" in activities:
        crowd_behaviour = "waiting"
    elif "walking" in activities or "walking together" in activities:
        crowd_behaviour = "walking"
    elif any(a.startswith("playing") for a in activities):
        crowd_behaviour = "playing"
    elif "working" in activities or "reading" in activities:
        crowd_behaviour = "working"
    elif "dining" in activities or "preparing food" in activities:
        crowd_behaviour = "eating"
    elif person_count >= 3:
        crowd_behaviour = "gathering"
    elif person_count == 0:
        crowd_behaviour = "none"

    emotional_atmosphere = "neutral"
    if env.crowd_level == "crowded":
        emotional_atmosphere = "busy"
    elif env.scene_complexity == "low" and person_count <= 1:
        emotional_atmosphere = "relaxed"
    elif "playing tennis" in activities or "playing with a ball" in activities:
        emotional_atmosphere = "competitive"
    elif env.social_context in {"group gathering", "small social interaction"}:
        emotional_atmosphere = "friendly"
    elif "working" in activities:
        emotional_atmosphere = "professional"

    scene_purpose = env.setting or env.scene_type
    environment_type = env.indoor_outdoor
    if env.setting in {"street", "parking lot", "road"}:
        environment_type = "urban"
    elif env.setting in {"park", "sports field", "tennis court", "beach"}:
        environment_type = "outdoor"
    elif env.setting in {"kitchen", "office", "classroom", "living room"}:
        environment_type = "indoor"

    return SceneEnrichment(
        age_group=age_group,
        crowd_behaviour=crowd_behaviour,
        emotional_atmosphere=emotional_atmosphere,
        scene_purpose=scene_purpose,
        environment_type=environment_type,
    )
