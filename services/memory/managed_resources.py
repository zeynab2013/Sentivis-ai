"""Managed resource scope and cleanup."""

from __future__ import annotations

from core.logging import get_logger
from services.interfaces.managed_resource import IManagedResource, IResourceScope

logger = get_logger(__name__)


class ResourceScope:
    """Tracks resources for a single pipeline stage."""

    def __init__(self) -> None:
        self._resources: list[IManagedResource] = []

    def register(self, resource: IManagedResource) -> None:
        """Register a resource for stage-end disposal."""
        self._resources.append(resource)

    def dispose_all(self) -> None:
        """Release and dispose all registered resources."""
        for resource in reversed(self._resources):
            if resource.is_acquired:
                resource.release()
            resource.dispose()
        self._resources.clear()
        logger.debug("Resource scope disposed")


class ManagedResourceManager:
    """Factory for resource scopes and emergency cleanup."""

    def create_scope(self) -> IResourceScope:
        """Create a new stage resource scope."""
        return ResourceScope()

    def force_dispose_all(self, resources: list[IManagedResource]) -> None:
        """Force cleanup after OOM or fatal error."""
        for resource in reversed(resources):
            try:
                if resource.is_acquired:
                    resource.release()
                resource.dispose()
            except RuntimeError as exc:
                logger.warning("Resource dispose error: %s", exc)
