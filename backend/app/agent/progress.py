"""Optional progress hooks for smoke scripts and debugging."""

from __future__ import annotations

import time
from collections.abc import Callable

_listeners: list[Callable[[str], None]] = []
_clock_start = time.perf_counter()


def reset_progress_clock() -> None:
    global _clock_start
    _clock_start = time.perf_counter()


def elapsed_seconds() -> float:
    return time.perf_counter() - _clock_start


def add_progress_listener(listener: Callable[[str], None]) -> None:
    _listeners.append(listener)


def clear_progress_listeners() -> None:
    _listeners.clear()


def report_progress(message: str) -> None:
    for listener in _listeners:
        listener(message)
