"""Pipeline cancellation token."""

from threading import Event

from core.exceptions.service import CancelledError


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation was requested."""
        if self.is_cancelled():
            raise CancelledError("Pipeline cancelled by user")
