"""Engine plugin registry."""

from collections.abc import Callable
from dataclasses import dataclass

from core.constants.pipeline_stages import PipelineStage
from core.logging import get_logger
from services.interfaces.model_engine import IModelEngine

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResourceRequirements:
    """Resource needs declared by a plugin."""

    min_vram_mb: float
    min_ram_mb: float
    preferred_device: str


@dataclass(frozen=True)
class PluginDescriptor:
    """Metadata for a registered engine plugin."""

    identifier: str
    version: str
    capabilities: tuple[str, ...]
    required_resources: ResourceRequirements
    supported_tasks: tuple[PipelineStage, ...]


class EnginePlugin:
    """Base plugin wrapping a model engine factory."""

    def __init__(
        self,
        descriptor: PluginDescriptor,
        factory: Callable[[], IModelEngine],
    ) -> None:
        self.descriptor = descriptor
        self._factory = factory

    def create_engine(self) -> IModelEngine:
        return self._factory()


class PluginRegistry:
    """Registry for swappable AI engine plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, EnginePlugin] = {}

    def register(self, plugin: EnginePlugin) -> None:
        self._plugins[plugin.descriptor.identifier] = plugin
        logger.info("Registered plugin %s", plugin.descriptor.identifier)

    def get(self, identifier: str) -> EnginePlugin:
        if identifier not in self._plugins:
            raise KeyError(f"Plugin not registered: {identifier}")
        return self._plugins[identifier]

    def list_plugins(self) -> tuple[PluginDescriptor, ...]:
        return tuple(plugin.descriptor for plugin in self._plugins.values())
