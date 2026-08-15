"""Register built-in engine plugins."""

from collections.abc import Callable

from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from services.interfaces.model_engine import IModelEngine
from services.plugins.plugin_registry import (
    EnginePlugin,
    PluginDescriptor,
    PluginRegistry,
    ResourceRequirements,
)


def register_builtin_plugins(
    plugin_registry: PluginRegistry,
    yolo_factory: Callable[[], IModelEngine],
    blip_factory: Callable[[], IModelEngine],
    gemma_factory: Callable[[], IModelEngine],
    florence_factory: Callable[[], IModelEngine] | None = None,
) -> None:
    """Register default Sentivis AI engine plugins."""
    plugin_registry.register(
        EnginePlugin(
            PluginDescriptor(
                identifier="vision.yolo_v8n",
                version="1.0.0",
                capabilities=("object_detection",),
                required_resources=ResourceRequirements(400, 2048, "cuda"),
                supported_tasks=(PipelineStage.YOLO_DETECTION,),
            ),
            yolo_factory,
        )
    )
    plugin_registry.register(
        EnginePlugin(
            PluginDescriptor(
                identifier="language.blip_base",
                version="1.0.0",
                capabilities=("vision_language",),
                required_resources=ResourceRequirements(1100, 4096, "cuda"),
                supported_tasks=(PipelineStage.BLIP_UNDERSTANDING,),
            ),
            blip_factory,
        )
    )
    if florence_factory is not None:
        plugin_registry.register(
            EnginePlugin(
                PluginDescriptor(
                    identifier="language.florence2",
                    version="1.0.0",
                    capabilities=("vision_language", "detailed_caption"),
                    required_resources=ResourceRequirements(900, 2048, "cuda"),
                    supported_tasks=(PipelineStage.BLIP_UNDERSTANDING,),
                ),
                florence_factory,
            )
        )
    plugin_registry.register(
        EnginePlugin(
            PluginDescriptor(
                identifier="language.gemma_2b",
                version="1.0.0",
                capabilities=("reasoning",),
                required_resources=ResourceRequirements(1100, 4096, "cuda"),
                supported_tasks=(PipelineStage.GEMMA_REASONING,),
            ),
            gemma_factory,
        )
    )


def wire_model_registry_from_plugins(
    plugin_registry: PluginRegistry,
    model_registry: object,
    plugin_ids: dict[ModelKind, str],
) -> None:
    """Bind model kinds to plugin factories via identifiers."""
    from services.models.model_manager import ModelRegistry

    assert isinstance(model_registry, ModelRegistry)
    mapping = {
        ModelKind.YOLO: plugin_ids.get(ModelKind.YOLO, "vision.yolo_v8n"),
        ModelKind.BLIP: plugin_ids.get(ModelKind.BLIP, "language.florence2"),
        ModelKind.GEMMA: plugin_ids.get(ModelKind.GEMMA, "language.gemma_2b"),
    }
    for kind, plugin_id in mapping.items():
        try:
            plugin = plugin_registry.get(plugin_id)
        except KeyError:
            if plugin_id == "language.florence2":
                plugin = plugin_registry.get("language.blip_base")
            else:
                raise
        model_registry.register(kind, plugin.create_engine)
