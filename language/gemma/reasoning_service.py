"""Model-backed reasoning service with evidence validation."""

from typing import Protocol

from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from core.contracts.analysis import SceneContext
from core.contracts.language import Prompt, RawCaption
from core.logging import get_logger
from language.validation.caption_validator import CaptionEvidenceValidator
from services.interfaces.model_manager import IModelManager

logger = get_logger(__name__)


class _ReasoningEngine(Protocol):
    def infer(self, prompt: Prompt) -> RawCaption:
        ...


class ManagedReasoningModel:
    """Runs Gemma reasoning through ModelManager and validates against evidence."""

    def __init__(self, model_manager: IModelManager, model_config: ModelConfig) -> None:
        self._model_manager = model_manager
        self._preferred_device = model_config.gemma.preferred_device
        self._validator = CaptionEvidenceValidator()

    def reason(self, prompt: Prompt, context: SceneContext) -> RawCaption:
        """Acquire Gemma, infer, validate, release."""
        engine = self._model_manager.acquire(ModelKind.GEMMA, self._preferred_device)
        try:
            reasoning: _ReasoningEngine = engine  # type: ignore[assignment]
            caption = reasoning.infer(prompt)
            validated = self._validator.filter_unsupported_sentences(caption.text, context)
            return RawCaption(
                text=validated,
                source=caption.source,
                confidence=caption.confidence,
            )
        finally:
            self._model_manager.release_active()
