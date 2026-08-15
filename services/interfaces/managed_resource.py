"""Managed resource interfaces."""

from typing import Protocol


class IManagedResource(Protocol):
    """Uniform resource lifecycle."""

    def initialize(self) -> None:
        ...

    def acquire(self) -> None:
        ...

    def release(self) -> None:
        ...

    def dispose(self) -> None:
        ...

    @property
    def memory_budget_mb(self) -> float:
        ...

    @property
    def is_acquired(self) -> bool:
        ...


class IResourceScope(Protocol):
    """Stage-scoped resource cleanup."""

    def register(self, resource: IManagedResource) -> None:
        ...

    def dispose_all(self) -> None:
        ...
