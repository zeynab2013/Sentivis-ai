"""Thread-local competition mode flag for deterministic pipeline behavior."""

from __future__ import annotations

import threading

_state = threading.local()


def activate(seed: int) -> None:
    """Enable competition mode for the current thread."""
    _state.active = True
    _state.seed = seed


def deactivate() -> None:
    """Disable competition mode for the current thread."""
    _state.active = False
    _state.seed = 0


def is_active() -> bool:
    return bool(getattr(_state, "active", False))


def seed() -> int:
    return int(getattr(_state, "seed", 0))
