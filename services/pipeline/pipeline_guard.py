"""Pre-stage and post-stage pipeline guards."""

from __future__ import annotations

import time

from core.config.app_config import AppConfig
from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.exceptions.service import PipelineTimeoutError
from core.logging import get_logger
from core.validation import output_validators
from services.memory.memory_guard import MemoryGuard
from services.models.device_selector import DeviceSelector
from services.models.model_validator import ModelValidator

logger = get_logger(__name__)

_GPU_STAGE_MODEL: dict[PipelineStage, ModelKind] = {
    PipelineStage.YOLO_DETECTION: ModelKind.YOLO,
    PipelineStage.BLIP_UNDERSTANDING: ModelKind.BLIP,
    PipelineStage.GEMMA_REASONING: ModelKind.GEMMA,
}


class PipelineGuard:
    """Validates prerequisites, memory, and outputs for pipeline stages."""

    def __init__(
        self,
        app_config: AppConfig,
        model_config: ModelConfig,
        memory_guard: MemoryGuard,
    ) -> None:
        self._timeout_seconds = app_config.hardware.pipeline_timeout_seconds
        self._started_at = 0.0
        self._memory_guard = memory_guard
        self._model_validator = ModelValidator(model_config, DeviceSelector(app_config))

    def begin_run(self) -> None:
        self._started_at = time.monotonic()

    def check_timeout(self) -> None:
        elapsed = time.monotonic() - self._started_at
        if elapsed > self._timeout_seconds:
            raise PipelineTimeoutError(self._timeout_seconds)

    def before_stage(self, stage: PipelineStage) -> None:
        self.check_timeout()
        model_kind = _GPU_STAGE_MODEL.get(stage)
        if model_kind is not None:
            self._model_validator.validate_for_kind(model_kind)
            self._memory_guard.ensure_stage_capacity(stage)

    def after_stage(self, stage: PipelineStage, output: object) -> None:
        self.check_timeout()
        if stage == PipelineStage.VALIDATION:
            output_validators.validate_validated_image(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.PREPROCESSING:
            output_validators.validate_preprocessed_image(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.YOLO_DETECTION:
            output_validators.validate_detection_result(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.SCENE_GRAPH:
            output_validators.validate_scene_graph(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.SCENE_CONTEXT:
            output_validators.validate_scene_context(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.BLIP_UNDERSTANDING and output is not None:
            output_validators.validate_visual_observations(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.PROMPT_BUILDING:
            output_validators.validate_prompt(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.CAPTION_REFINEMENT:
            output_validators.validate_refined_caption(output)  # type: ignore[arg-type]
        elif stage == PipelineStage.QUALITY_EVALUATION:
            output_validators.validate_quality_report(output)  # type: ignore[arg-type]
        logger.debug("Stage %s output validated", stage.name)
