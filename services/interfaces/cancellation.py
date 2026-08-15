"""Cancellation token interface."""

from typing import Protocol


class ICancellationToken(Protocol):
    """Cooperative pipeline cancellation."""

    def cancel(self) -> None:
        ...

    def is_cancelled(self) -> bool:
        ...

    def raise_if_cancelled(self) -> None:
        ...
