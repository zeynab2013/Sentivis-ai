"""Fallback captions derived from structured scene evidence."""

from __future__ import annotations

from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption

_GENERIC_ACTIVITIES = frozenset(
    {
        "static scene",
        "people present",
        "waiting",
        "having a conversation",
        "transportation scene",
        "standing",
        "sitting",
    }
)
_PLACEHOLDER_SETTINGS = frozenset(
    {
        "",
        "unknown",
        "general",
        "general scene",
        "photographed scene",
        "everyday environment",
    }
)


def build_context_caption(context: SceneContext) -> RawCaption:
    """Build a conservative natural-language caption from graph evidence.

    Never emits internal pipeline labels such as ``Observed activity:`` or
    inventory stubs like ``Person, and bicycle. The location is outdoor.``
    """
    if not context.graph.nodes:
        return RawCaption(
            text="The image content could not be described with sufficient confidence.",
            source="context",
            confidence=0.45,
        )

    env = context.environment
    labels = [node.label.strip().lower() for node in context.graph.nodes[:6] if node.label]
    people_n = sum(1 for lab in labels if lab == "person")
    objects = [lab for lab in labels if lab != "person"]

    meaningful_activities = [
        item
        for item in context.activities.activities
        if item.activity.lower() not in _GENERIC_ACTIVITIES and item.confidence >= 0.65
    ]

    place = ""
    if env.indoor_outdoor not in _PLACEHOLDER_SETTINGS:
        io = env.indoor_outdoor.strip().lower()
        if io in {"outdoor", "outdoors"}:
            place = "outdoors"
        elif io in {"indoor", "indoors"}:
            place = "indoors"
        else:
            place = io
    elif env.setting not in _PLACEHOLDER_SETTINGS:
        place = env.setting.strip().lower()

    if meaningful_activities:
        activity = meaningful_activities[0].activity.strip()
        # Prefer a natural progressive sentence from the verified activity.
        if people_n >= 1:
            lead = "A person is" if people_n == 1 else "People are"
            text = f"{lead} {activity}"
        else:
            text = activity[:1].upper() + activity[1:] if activity else ""
        if place and place not in text.lower():
            text = f"{text} {place}"
        text = text.rstrip(".") + "."
        return RawCaption(text=text, source="context", confidence=0.65)

    # No strong activity — short natural presence sentence (never inventory lists).
    if people_n >= 1 and objects:
        obj = objects[0]
        article = "an" if obj[:1] in "aeiou" else "a"
        who = "A person" if people_n == 1 else f"{people_n} people"
        text = f"{who} and {article} {obj} are visible"
    elif people_n >= 1:
        text = "A person is visible" if people_n == 1 else f"{people_n} people are visible"
    elif objects:
        obj = objects[0]
        article = "an" if obj[:1] in "aeiou" else "a"
        text = f"{article.capitalize()} {obj} is visible"
    else:
        text = "The image shows verified objects with limited additional context"
    if place and place not in text.lower():
        text = f"{text} {place}"
    text = text.rstrip(".") + "."
    return RawCaption(text=text, source="context", confidence=0.65)
