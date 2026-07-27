"""Discrete-event scheduling (manuscript Section 15): a time-ordered event
queue, the standard mechanism by which a discrete-event simulation
advances -- events are processed strictly in non-decreasing time order,
with ties broken by scheduling (insertion) order for determinism.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digital_twin.events import SimulationEvent


class EventQueue:
    """A min-heap ordered by (event.time, insertion_sequence). The
    insertion-sequence tiebreaker is what makes iteration order
    deterministic for two events scheduled at the same simulation time --
    without it, Python's heap would fall back to comparing `SimulationEvent`
    instances directly, which is both unnecessary and undefined for
    dataclasses that don't define ordering.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, SimulationEvent]] = []
        self._counter = itertools.count()

    def schedule(self, event: SimulationEvent) -> None:
        heapq.heappush(self._heap, (event.time, next(self._counter), event))

    def peek_next_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def pop_next(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("pop_next() called on an empty EventQueue")
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
