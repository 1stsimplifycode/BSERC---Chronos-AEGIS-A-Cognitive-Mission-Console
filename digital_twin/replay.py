"""Deterministic replay (manuscript Section 15's implicit reproducibility
requirement, made explicit for this phase's "allow experiments to replay
identical scenarios" goal).

Every `SimulationEvent` is an immutable, fully-specified dataclass
(`digital_twin/events.py`) scheduled up front from a `ScenarioSpec`
(`digital_twin/scenarios.py`); the only source of non-determinism a twin
run could introduce is its seeded RNG (`MissionDigitalTwin.rng`,
`node_feature_vector`). Consequently "replay" here means: given the same
scenario spec and the same seed, re-running produces byte-for-byte the
same applied-event sequence and the same feature vectors -- verified
explicitly by `replay`, not merely assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from digital_twin.scenarios import ScenarioSpec
from digital_twin.twin import MissionDigitalTwin


class ReplayMismatchError(RuntimeError):
    """Raised when replaying a scenario with the same seed produces a
    different event trajectory than the original run -- a genuine
    determinism failure, never silently ignored."""


@dataclass(frozen=True)
class ReplayLog:
    """A record of one scenario run, sufficient to verify a later run
    reproduces it exactly."""

    scenario_name: str
    seed: int
    applied_events: tuple[object, ...]
    final_outcome: object | None
    final_phase: str
    node_feature_fingerprints: tuple[tuple[str, tuple[float, ...]], ...] = field(
        default_factory=tuple
    )


def run_scenario(spec: ScenarioSpec, seed: int = 0) -> tuple[MissionDigitalTwin, ReplayLog]:
    """Schedule every event in `spec` and run the twin to completion,
    returning both the twin and a `ReplayLog` capturing what happened."""
    twin = MissionDigitalTwin(spec.name, seed=seed)
    for event in spec.events:
        twin.schedule(event)
    twin.run()
    fingerprints = tuple(
        (node_id, tuple(twin.node_feature_vector(node_id).tolist()))
        for node_id in sorted(twin.graph.nodes)
    )
    log = ReplayLog(
        scenario_name=spec.name,
        seed=seed,
        applied_events=tuple(twin.event_log),
        final_outcome=twin.env.outcome,
        final_phase=twin.env.phase,
        node_feature_fingerprints=fingerprints,
    )
    return twin, log


def replay(spec: ScenarioSpec, log: ReplayLog) -> MissionDigitalTwin:
    """Re-run `spec` with `log.seed` and verify the resulting trajectory
    matches `log` exactly -- applied events, final outcome/phase, and every
    node's deterministic feature vector. Raises `ReplayMismatchError` on
    any divergence rather than silently returning a possibly-different run.
    """
    twin, new_log = run_scenario(spec, seed=log.seed)
    if new_log.applied_events != log.applied_events:
        raise ReplayMismatchError(
            f"replay of scenario {spec.name!r} (seed={log.seed}) diverged: "
            f"applied event sequence does not match the original run"
        )
    if new_log.final_outcome != log.final_outcome or new_log.final_phase != log.final_phase:
        raise ReplayMismatchError(
            f"replay of scenario {spec.name!r} (seed={log.seed}) diverged: "
            f"final outcome/phase does not match the original run"
        )
    if new_log.node_feature_fingerprints != log.node_feature_fingerprints:
        raise ReplayMismatchError(
            f"replay of scenario {spec.name!r} (seed={log.seed}) diverged: "
            f"node feature vectors do not match the original run"
        )
    return twin
