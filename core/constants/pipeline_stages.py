"""Pipeline stage enumeration and ordering."""

from enum import Enum, auto


class PipelineStage(Enum):
    """Ordered pipeline stages for visual understanding."""

    LOAD_IMAGE = auto()
    VALIDATION = auto()
    PREPROCESSING = auto()
    YOLO_DETECTION = auto()
    ATTRIBUTE_EXTRACTION = auto()
    RELATIONSHIP_ANALYSIS = auto()
    SCENE_GRAPH = auto()
    ACTIVITY_ANALYSIS = auto()
    SCENE_CONTEXT = auto()
    BLIP_UNDERSTANDING = auto()
    PROMPT_BUILDING = auto()
    GEMMA_REASONING = auto()
    CAPTION_REFINEMENT = auto()
    QUALITY_EVALUATION = auto()
    EXPORT = auto()

    @classmethod
    def ordered_stages(cls) -> tuple["PipelineStage", ...]:
        """Return pipeline stages in execution order excluding load/export helpers."""
        return (
            cls.VALIDATION,
            cls.PREPROCESSING,
            cls.YOLO_DETECTION,
            cls.ATTRIBUTE_EXTRACTION,
            cls.RELATIONSHIP_ANALYSIS,
            cls.SCENE_GRAPH,
            cls.ACTIVITY_ANALYSIS,
            cls.SCENE_CONTEXT,
            cls.BLIP_UNDERSTANDING,
            cls.PROMPT_BUILDING,
            cls.GEMMA_REASONING,
            cls.CAPTION_REFINEMENT,
            cls.QUALITY_EVALUATION,
        )

    @property
    def display_name(self) -> str:
        """Human-readable stage label for UI progress."""
        return _DISPLAY_NAMES[self]


_DISPLAY_NAMES: dict[PipelineStage, str] = {
    PipelineStage.LOAD_IMAGE: "Loading Image",
    PipelineStage.VALIDATION: "Validating Image",
    PipelineStage.PREPROCESSING: "Preprocessing",
    PipelineStage.YOLO_DETECTION: "Object Detection",
    PipelineStage.ATTRIBUTE_EXTRACTION: "Extracting Attributes",
    PipelineStage.RELATIONSHIP_ANALYSIS: "Analyzing Relationships",
    PipelineStage.SCENE_GRAPH: "Building Scene Graph",
    PipelineStage.ACTIVITY_ANALYSIS: "Analyzing Activities",
    PipelineStage.SCENE_CONTEXT: "Building Scene Context",
    PipelineStage.BLIP_UNDERSTANDING: "Visual Understanding",
    PipelineStage.PROMPT_BUILDING: "Building Prompt",
    PipelineStage.GEMMA_REASONING: "Reasoning",
    PipelineStage.CAPTION_REFINEMENT: "Refining Caption",
    PipelineStage.QUALITY_EVALUATION: "Evaluating Caption Quality",
    PipelineStage.EXPORT: "Exporting",
}
