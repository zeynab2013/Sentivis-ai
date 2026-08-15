"""Timing utilities."""

import time
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def stopwatch() -> Generator[list[float], None, None]:
    """Context manager recording elapsed seconds in a one-element list.

    Yields:
        List whose index 0 is updated with elapsed seconds on exit.
    """
    elapsed: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start
