"""Cross-module data transfer objects."""

from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    Attribute,
    AttributeSet,
    EnvironmentInfo,
    Relation,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.detection import BoundingBox, Detection, DetectionResult, SegmentationMask
from core.contracts.image import ImagePayload, PreprocessedImage, ValidatedImage
from core.contracts.language import (
    CaptionQualityReport,
    Prompt,
    RawCaption,
    RefinedCaption,
    VisualObservations,
)
from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult, StageProgress

__all__ = [
    "ActivityEvidence",
    "ActivityHints",
    "CaptionQualityReport",
    "AnalysisOptions",
    "Attribute",
    "AttributeSet",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "EnvironmentInfo",
    "ImagePayload",
    "PipelineRequest",
    "PipelineResult",
    "PreprocessedImage",
    "Prompt",
    "RawCaption",
    "RefinedCaption",
    "Relation",
    "SceneContext",
    "SceneGraph",
    "SceneNode",
    "SegmentationMask",
    "StageProgress",
    "ValidatedImage",
    "VisualObservations",
]
