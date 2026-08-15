"""Analysis heuristics configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttributeHeuristicsConfig:
    """Size, distance, pose, and visibility thresholds."""

    size_small_max_ratio: float
    size_medium_max_ratio: float
    zone_split_low: float
    zone_split_high: float
    distance_near_ratio: float
    distance_medium_ratio: float
    pose_standing_ratio: float
    pose_lying_ratio: float
    visibility_high_threshold: float
    visibility_medium_threshold: float


@dataclass(frozen=True)
class RelationshipHeuristicsConfig:
    """Spatial relationship inference thresholds."""

    overlap_distance: float
    overlap_confidence: float
    distance_confidence_factor: float
    max_confidence: float
    near_distance_ratio: float
    far_distance_ratio: float


@dataclass(frozen=True)
class ActivityHeuristicsConfig:
    """Activity inference confidence values."""

    confidence_with_nodes: float
    confidence_empty: float


@dataclass(frozen=True)
class ActivityReasoningConfig:
    """Legacy activity reasoning flags (activities are heuristic-only)."""

    enabled: bool
    mode: str
    model: str
    base_url: str
    timeout_seconds: float
    fallback_to_minimal: bool
    prefer_ollama_caption: bool
    models: tuple[str, ...]


@dataclass(frozen=True)
class SemanticReasoningConfig:
    """Ollama high-level semantic synthesis configuration."""

    enabled: bool
    mode: str
    model: str
    base_url: str
    timeout_seconds: float
    fallback_to_context_caption: bool
    prefer_over_gemma: bool
    models: tuple[str, ...]


@dataclass(frozen=True)
class ContextHeuristicsConfig:
    """Scene context inference thresholds."""

    crowd_threshold: int
    complexity_high_relations: int
    complexity_medium_relations: int


@dataclass(frozen=True)
class AnalysisConfig:
    """Aggregate analysis heuristics configuration."""

    attributes: AttributeHeuristicsConfig
    relationships: RelationshipHeuristicsConfig
    activity: ActivityHeuristicsConfig
    activity_reasoning: ActivityReasoningConfig
    semantic_reasoning: SemanticReasoningConfig
    context: ContextHeuristicsConfig

    def size_labels(self) -> tuple[tuple[float, str], ...]:
        return (
            (self.attributes.size_small_max_ratio, "small"),
            (self.attributes.size_medium_max_ratio, "medium"),
            (1.0, "large"),
        )
