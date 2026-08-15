"""Model manager interface."""

from typing import Protocol

from core.constants.model_kinds import ModelKind
from services.interfaces.model_engine import IModelEngine


class IModelManager(Protocol):
    """Sole authority for model load and unload."""

    def acquire(self, kind: ModelKind, preferred_device: str) -> IModelEngine:
        ...

    def release_active(self) -> None:
        ...

    def release_all(self) -> None:
        ...
