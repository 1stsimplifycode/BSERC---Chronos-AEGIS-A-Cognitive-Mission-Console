"""The simulation clock (manuscript Section 15): a monotonic notion of
simulation time the Digital Twin's discrete-event engine advances as
events are applied, independent of wall-clock time so scenarios are
reproducible regardless of how long they take to execute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationClock:
    """Monotonic simulation time. `advance_to` is the only way time moves
    forward -- it is a programming error for an event to be applied out of
    time order, so this raises rather than silently reordering."""

    _now: float = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance_to(self, t: float) -> None:
        if t < self._now:
            raise ValueError(f"cannot advance simulation clock backwards: {t} < {self._now}")
        self._now = t
