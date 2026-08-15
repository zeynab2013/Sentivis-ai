"""Application dependency injection container."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from analysis.activity.activity_reasoning_service import ActivityReasoningService
from analysis.attributes.attribute_extractor import AttributeExtractor
from analysis.context.context_builder import ContextBuilder
from analysis.ocr.text_extractor import OcrExtractor
from analysis.pose.pose_estimator import PoseEstimator
from analysis.relationships.relationship_analyzer import RelationshipAnalyzer
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from analysis.scene_reasoner.scene_reasoner import SceneReasoner
from analysis.semantic.semantic_reasoning_service import SemanticReasoningService
from app.plugin_bootstrap import register_builtin_plugins, wire_model_registry_from_plugins
from core.config.analysis_config import AnalysisConfig
from core.config.app_config import AppConfig
from core.config.model_config import ModelConfig
from core.config.theme_config import ThemeConfig
from core.constants.model_kinds import ModelKind
from language.blip.blip_engine import BlipEngine
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.florence.florence_engine import FlorenceEngine
from language.gemma.gemma_engine import GemmaEngine
from language.gemma.reasoning_service import ManagedReasoningModel
from language.prompts.prompt_builder import PromptBuilder
from language.refinement.caption_refiner import CaptionRefiner
from language.semantic.natural_caption_service import NaturalCaptionService
from language.vlm.managed_vision_model import ManagedVisionModel
from release.metadata import ReleaseInfo, load_release_info
from services.cache.cache_manager import CacheManager
from services.export.export_manager import ExportManager
from services.memory.memory_guard import MemoryGuard
from services.memory.memory_manager import MemoryManager
from services.models.device_selector import DeviceSelector
from services.models.model_manager import ModelManager, ModelRegistry
from services.pipeline.cancellation import CancellationToken
from services.pipeline.metrics_collector import PipelineMetricsCollector
from services.pipeline.orchestrator import PipelineOrchestrator
from services.pipeline.pipeline_guard import PipelineGuard
from services.pipeline.progress_reporter import ProgressReporter
from services.pipeline.quality_assurance import PipelineQualityAssurance
from services.pipeline.stage_runner import StageRunner
from services.plugins.plugin_registry import PluginRegistry
from services.runtime.assets import build_runtime_assets
from services.runtime.cache_maintenance import CacheMaintenanceService
from services.runtime.model_discovery import resolve_model_search_paths
from services.runtime.model_registry import CentralModelRegistry
from services.runtime.model_validation import ModelValidationService
from services.runtime.self_test import SelfTestRunner
from services.runtime.status_provider import RuntimeStatusProvider
from ui.application_facade import ApplicationFacade
from ui.controllers.export_controller import ExportController
from ui.controllers.main_controller import MainController
from ui.controllers.pipeline_controller import PipelineController
from ui.controllers.settings_controller import SettingsController
from ui.themes.theme_manager import ThemeManager
from ui.view_models.export_view_model import ExportViewModel
from ui.view_models.history_view_model import HistoryViewModel
from ui.view_models.pipeline_view_model import PipelineViewModel
from ui.view_models.settings_view_model import SettingsViewModel
from vision.detection.detector_service import ManagedObjectDetector
from vision.detection.refining_detector import RefiningObjectDetector
from vision.detection.yolo_engine import YoloEngine
from vision.preprocessing.enhanced_preprocessor import EnhancedPreprocessor
from vision.validation.image_validator import ImageValidator

if TYPE_CHECKING:
    from model_management.service import ModelManagementService


@dataclass
class ApplicationContext:
    """Wired application services for lifecycle management."""

    facade: ApplicationFacade
    main_controller: MainController
    model_manager: ModelManager
    memory_manager: MemoryManager
    plugin_registry: PluginRegistry
    runtime_status: RuntimeStatusProvider
    model_registry: CentralModelRegistry
    release_info: ReleaseInfo
    model_management: ModelManagementService | None = field(default=None)


class DependencyContainer:
    """Builds and binds interface implementations."""

    def build(
        self,
        app_config: AppConfig,
        model_config: ModelConfig,
        theme_config: ThemeConfig,
        analysis_config: AnalysisConfig,
    ) -> ApplicationContext:
        release_info = load_release_info()
        app_config = replace(app_config, app_version=release_info.display_version)
        memory_manager = MemoryManager(app_config)
        device_selector = DeviceSelector(app_config)

        yolo_search_paths = resolve_model_search_paths(
            app_config.paths.models_dir,
            app_config.paths.model_search_paths,
        )

        def yolo_factory() -> YoloEngine:
            return YoloEngine(model_config.yolo, search_paths=yolo_search_paths)

        def blip_factory() -> BlipEngine:
            return BlipEngine(model_config.blip)

        def florence_factory() -> FlorenceEngine:
            return FlorenceEngine(model_config.florence, model_config.blip)

        def gemma_factory() -> GemmaEngine:
            return GemmaEngine(model_config.gemma)

        plugin_registry = PluginRegistry()
        register_builtin_plugins(
            plugin_registry,
            yolo_factory,
            blip_factory,
            gemma_factory,
            florence_factory=florence_factory,
        )

        model_registry = ModelRegistry()
        wire_model_registry_from_plugins(
            plugin_registry,
            model_registry,
            {
                ModelKind.YOLO: model_config.plugins.detection_plugin,
                ModelKind.BLIP: model_config.plugins.vision_language_plugin,
                ModelKind.GEMMA: model_config.plugins.reasoning_plugin,
            },
        )

        model_validator = ModelValidationService(model_config, device_selector)
        central_registry = CentralModelRegistry(
            app_config,
            model_config,
            plugin_registry,
            model_validator,
            extra_search_paths=app_config.paths.model_search_paths,
        )
        model_manager = ModelManager(
            model_registry,
            memory_manager,
            device_selector,
            model_catalog=central_registry,
        )

        progress = ProgressReporter()
        cancellation = CancellationToken()
        memory_guard = MemoryGuard(app_config, memory_manager)
        pipeline_guard = PipelineGuard(app_config, model_config, memory_guard)
        metrics_collector = PipelineMetricsCollector(memory_manager)
        quality_assurance = PipelineQualityAssurance(app_config.competition)
        stage_runner = StageRunner(
            progress,
            cancellation,
            memory_manager,
            pipeline_guard,
            metrics_collector,
            model_manager,
        )

        base_detector = ManagedObjectDetector(model_manager, model_config)
        detector = RefiningObjectDetector(base_detector, app_config, model_config)
        managed_vlm = ManagedVisionModel(model_config, model_config.vlm)
        caption_refiner = CaptionRefiner()
        quality_evaluator = CaptionQualityEvaluator()
        natural_caption = NaturalCaptionService(
            managed_vlm,
            evaluator=quality_evaluator,
            refiner=caption_refiner,
        )
        orchestrator = PipelineOrchestrator(
            validator=ImageValidator(app_config),
            preprocessor=EnhancedPreprocessor(app_config),
            detector=detector,
            attribute_extractor=AttributeExtractor(analysis_config),
            relationship_analyzer=RelationshipAnalyzer(analysis_config),
            scene_graph_builder=SceneGraphBuilder(analysis_config),
            activity_reasoning=ActivityReasoningService(analysis_config),
            semantic_reasoning=SemanticReasoningService(analysis_config),
            context_builder=ContextBuilder(analysis_config),
            vision_language=managed_vlm,
            prompt_builder=PromptBuilder(),
            reasoning_model=ManagedReasoningModel(model_manager, model_config),
            caption_refiner=caption_refiner,
            quality_evaluator=quality_evaluator,
            stage_runner=stage_runner,
            model_manager=model_manager,
            memory_manager=memory_manager,
            progress=progress,
            app_config=app_config,
            metrics_collector=metrics_collector,
            scene_reasoner=SceneReasoner(),
            natural_caption=natural_caption,
            pose_estimator=PoseEstimator(),
            ocr_extractor=OcrExtractor(),
            quality_assurance=quality_assurance,
            pipeline_guard=pipeline_guard,
        )
        export_manager = ExportManager()
        cache_manager = CacheManager(app_config)
        runtime_assets = build_runtime_assets(app_config)
        for manager in runtime_assets.all_managers():
            manager.ensure_directory()
        cache_maintenance = CacheMaintenanceService(app_config, cache_manager)
        self_test_runner = SelfTestRunner(
            app_config,
            central_registry,
            runtime_assets,
            plugin_registry,
            cache_maintenance,
        )
        runtime_status = RuntimeStatusProvider(
            central_registry,
            cache_maintenance,
            self_test_runner,
        )
        runtime_status.latest_self_test()

        pipeline_controller = PipelineController(orchestrator, progress)
        export_controller = ExportController(export_manager, app_config)
        settings_controller = SettingsController(app_config, theme_config)

        pipeline_view_model = PipelineViewModel(pipeline_controller)
        export_view_model = ExportViewModel(export_controller, pipeline_view_model)
        history_view_model = HistoryViewModel()
        theme_manager = ThemeManager(theme_config)
        settings_view_model = SettingsViewModel(settings_controller, theme_manager)
        facade = ApplicationFacade(
            app_config,
            theme_config,
            theme_manager,
            pipeline_view_model,
            export_view_model,
            history_view_model,
            settings_view_model,
        )

        main_controller = MainController(
            app_config=app_config,
            pipeline_controller=pipeline_controller,
            export_controller=export_controller,
            settings_controller=settings_controller,
            model_manager=model_manager,
            memory_manager=memory_manager,
            cache_manager=cache_manager,
        )
        return ApplicationContext(
            facade=facade,
            main_controller=main_controller,
            model_manager=model_manager,
            memory_manager=memory_manager,
            plugin_registry=plugin_registry,
            runtime_status=runtime_status,
            model_registry=central_registry,
            release_info=release_info,
        )
