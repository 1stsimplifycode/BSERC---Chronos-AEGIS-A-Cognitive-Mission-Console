"""The Mission Digital Twin (manuscript Section 15).

"We retain, unchanged, the mission digital twin from our preceding
formulation: a simulation environment spanning aircraft subsystems,
software, firmware, sensors, networks, communications, weather, and
injected cyber events, serving as the shared substrate on which every CMI
component is trained and validated. We extend its role in two respects
specific to CMI. First, the twin is now also the source of the
ground-truth outcome records o that populate Episodic Memory (Section
8.2.3)... Second, the twin's capacity for controlled intervention --
injecting a specific, known perturbation with known downstream effect --
is what Section 10.4 identifies as the substrate on which causal
discovery for the TRMG's causal-graph subset would be performed."

`MissionDigitalTwin` is a discrete-event engine: it owns a real
`TemporalReasoningMemoryGraph` (the same graph every other CMI component
reads from -- Section 6, unchanged) and a `MissionEnvironment` (state.py),
and advances a `SimulationClock` by popping and applying scheduled
`SimulationEvent`s in time order. It is a producer, not a consumer: it
never calls into `reasoning/`, `memory/`, `agents/`, `cyber/`, or
`scheduler/` itself (see `digital_twin/integration.py` for where the
resulting TRMG is handed to `TTRNPipeline`, unchanged, exactly as it would
be handed a hand-built graph in `examples/`).
"""

from __future__ import annotations

import hashlib

import numpy as np

from digital_twin.events import SimulationEvent
from digital_twin.queue import EventQueue
from digital_twin.state import MissionEnvironment
from graph.trmg.model import TemporalReasoningMemoryGraph


class MissionDigitalTwin:
    """A discrete-event simulation of one mission (Section 15)."""

    def __init__(self, mission_id: str, seed: int = 0) -> None:
        self.graph = TemporalReasoningMemoryGraph()
        self.env = MissionEnvironment(mission_id=mission_id)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._queue = EventQueue()
        self.event_log: list[SimulationEvent] = []
        """Every event actually applied, in application order -- the
        replay log `digital_twin/replay.py` uses to verify determinism."""

    def schedule(self, event: SimulationEvent) -> None:
        """Add `event` to the queue. Does not apply it -- events are only
        applied in time order via `run`/`step` (discrete-event semantics:
        scheduling is not the same as happening)."""
        if event.time < self.env.time:
            raise ValueError(
                f"cannot schedule event at t={event.time} before current simulation "
                f"time t={self.env.time}"
            )
        self._queue.schedule(event)

    def pending_event_count(self) -> int:
        return len(self._queue)

    def step(self) -> SimulationEvent | None:
        """Apply exactly the next scheduled event, advancing the clock to
        its time. Returns None (and does nothing) if the queue is empty."""
        if not self._queue:
            return None
        event = self._queue.pop_next()
        self.env.time = event.time
        event.apply(self)
        self.event_log.append(event)
        return event

    def run(self, until: float | None = None) -> int:
        """Apply scheduled events in time order until either the queue is
        empty or the next event's time would exceed `until` (exclusive of
        events scheduled to run *at* `until`, which are applied). Returns
        the number of events applied.

        `until=None` (the default) drains the queue completely -- runs the
        scenario to its end.
        """
        applied = 0
        while True:
            next_time = self._queue.peek_next_time()
            if next_time is None:
                break
            if until is not None and next_time > until:
                break
            self.step()
            applied += 1
        return applied

    def node_feature_vector(self, node_id: str, dim: int = 8) -> np.ndarray:
        """A deterministic, reproducible feature vector for `node_id`,
        derived from this twin's seeded RNG stream keyed by the node's
        identity -- not a claim of realistic sensor-derived features (this
        twin has no physical sensor model), but a documented, seed-stable
        bridge so `reasoning.trust.gnn.MissionAwareGNN` (itself untrained,
        see its own module docstring) has *some* deterministic input to
        attend over. `digital_twin/integration.py` uses this as its
        default; callers with a richer feature model may supply their own.

        PHASE 8 FIX: the node seed is derived via `hashlib.sha256`, not
        Python's built-in `hash()`. `hash()` on a `str` (directly or
        nested in a tuple, as this method originally did with
        `hash((seed, node_id))`) is salted per-process by
        `PYTHONHASHSEED` (Python's hash-flooding DoS mitigation, on by
        default since Python 3.3): the *same* seed and node_id produced a
        *different* feature vector in every separate process invocation,
        silently breaking cross-process reproducibility while still
        appearing perfectly deterministic within any single process (e.g.
        an entire pytest run, or one REPL session) -- which is exactly why
        Phase 7's determinism tests (`tests/digital_twin/test_twin.py`,
        `digital_twin/replay.py`) never caught it: pytest runs all tests
        in one process, so `hash()` was consistent for the whole suite.
        This was only discovered in Phase 8 by hashing
        `scripts/reproduce_all.py`'s output across two separate `python`
        invocations with the same seeds and finding it did not match.
        `hashlib.sha256` is not affected by `PYTHONHASHSEED` and gives the
        same digest for the same input in every process, on every run,
        forever.
        """
        digest = hashlib.sha256(f"{self.seed}:{node_id}".encode()).digest()
        node_seed = int.from_bytes(digest[:4], byteorder="big")
        return np.random.default_rng(node_seed).normal(size=dim)
