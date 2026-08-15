"""Application-wide constants."""

from core.constants.limits import (
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    YOLO_INFERENCE_SIZE,
)
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage

__all__ = [
    "MAX_FILE_SIZE_BYTES",
    "MAX_IMAGE_DIMENSION",
    "ModelKind",
    "PipelineStage",
    "YOLO_INFERENCE_SIZE",
]
