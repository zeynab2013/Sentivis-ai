"""Model engine lifecycle interface."""

from typing import Protocol

from core.constants.model_kinds import ModelKind


class IModelEngine(Protocol):
    """Common heavy-model lifecycle contract."""

    @property
    def model_kind(self) -> ModelKind:
        ...

    @property
    def is_loaded(self) -> bool:
        ...

    @property
    def device(self) -> str:
        ...

    def set_device(self, device: str) -> None:
        ...

    def load(self) -> None:
        ...

    def release(self) -> None:
        ...

    def clear_device_cache(self) -> None:
        ...
