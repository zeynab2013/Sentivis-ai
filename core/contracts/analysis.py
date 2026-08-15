"""Scene analysis DTOs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute:
    """Attribute assigned to a detected object."""

    object_index: int
    name: str
    value: str


@dataclass(frozen=True)
class AttributeSet:
    """Collection of object attributes."""

    attributes: tuple[Attribute, ...]


@dataclass(frozen=True)
class Relation:
    """Spatial or semantic relation between two objects."""

    subject_index: int
    object_index: int
    relation_type: str
    confidence: float


@dataclass(frozen=True)
class SceneNode:
    """Node in a scene graph representing one object."""

    index: int
    object_id: str
    label: str
    bounding_box_area_ratio: float
    position_zone: str


@dataclass(frozen=True)
class SceneGraph:
    """Graph of objects and their relations."""

    nodes: tuple[SceneNode, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class ActivityEvidence:
    """Evidence-backed activity inference."""

    activity: str
    confidence: float
    supporting_node_indices: tuple[int, ...]
    supporting_relation_types: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ActivityHints:
    """Inferred human activities from scene graph structure."""

    activities: tuple[ActivityEvidence, ...]
    confidence: float


@dataclass(frozen=True)
class EnvironmentInfo:
    """Environmental context of the scene."""

    scene_type: str
    setting: str
    time_of_day: str
    weather: str
    indoor_outdoor: str
    social_context: str
    crowd_level: str
    scene_complexity: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SceneContext:
    """Aggregated structured scene understanding."""

    graph: SceneGraph
    attributes: AttributeSet
    activities: ActivityHints
    environment: EnvironmentInfo
    object_count: int
    dominant_objects: tuple[str, ...]
    spatial_summary: str
