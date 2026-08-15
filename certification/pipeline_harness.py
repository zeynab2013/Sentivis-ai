"""Pipeline harness for certification workflows (no test-package dependency)."""

from __future__ import annotations

from analysis.activity.activity_reasoning_service import ActivityReasoningService
from analysis.attributes.attribute_extractor import AttributeExtractor
from analysis.context.context_builder import ContextBuilder
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from analysis.semantic.semantic_reasoning_service import SemanticReasoningService
from core.config.loader import load_analysis_config, load_app_config, load_model_config
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.interfaces.reasoning import IReasoningModel
from language.interfaces.vision_language import IVisionLanguageModel
from language.prompts.prompt_builder import PromptBuilder
from language.refinement.caption_refiner import CaptionRefiner
from services.memory.memory_manager import MemoryManager
from services.models.device_selector import DeviceSelector
from services.models.model_manager import ModelManager, ModelRegistry
from services.pipeline.cancellation import CancellationToken
from services.pipeline.metrics_collector import PipelineMetricsCollector
from services.pipeline.orchestrator import PipelineOrchestrator
from services.pipeline.progress_reporter import ProgressReporter
from services.pipeline.quality_assurance import PipelineQualityAssurance
from services.pipeline.stage_runner import StageRunner
from vision.interfaces.detector import IObjectDetector
from vision.preprocessing.standard_preprocessor import StandardPreprocessor
from vision.validation.image_validator import ImageValidator


def build_test_orchestrator(
    detector: IObjectDetector,
    vision_language: IVisionLanguageModel,
    reasoning_model: IReasoningModel,
) -> PipelineOrchestrator:
    """Construct a pipeline orchestrator wired with injectable model stubs."""
    app_config = load_app_config()
    analysis_config = load_analysis_config()
    load_model_config()
    memory_manager = MemoryManager(app_config)
    model_manager = ModelManager(ModelRegistry(), memory_manager, DeviceSelector(app_config))
    progress = ProgressReporter()
    metrics_collector = PipelineMetricsCollector(memory_manager)
    quality_assurance = PipelineQualityAssurance(app_config.competition)
    stage_runner = StageRunner(
        progress,
        CancellationToken(),
        memory_manager,
        None,
        metrics_collector,
        model_manager,
    )
    return PipelineOrchestrator(
        validator=ImageValidator(app_config),
        preprocessor=StandardPreprocessor(app_config),
        detector=detector,
        attribute_extractor=AttributeExtractor(analysis_config),
        relationship_analyzer=RelationshipAnalyzer(analysis_config),
        scene_graph_builder=SceneGraphBuilder(analysis_config),
        activity_reasoning=ActivityReasoningService(analysis_config),
        semantic_reasoning=SemanticReasoningService(analysis_config),
        context_builder=ContextBuilder(analysis_config),
        vision_language=vision_language,
        prompt_builder=PromptBuilder(),
        reasoning_model=reasoning_model,
        caption_refiner=CaptionRefiner(),
        quality_evaluator=CaptionQualityEvaluator(),
        stage_runner=stage_runner,
        model_manager=model_manager,
        memory_manager=memory_manager,
        progress=progress,
        app_config=app_config,
        metrics_collector=metrics_collector,
        quality_assurance=quality_assurance,
    )
