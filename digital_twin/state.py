"""Twin-level mission state (manuscript Section 15) that has no TRMG-node
analogue.

Everything Section 15 lists as simulated -- "aircraft state, mission
phases, telemetry, communications, software components, firmware,
sensors, maintenance state, weather, cyber incidents, trust evolution" --
already has a home as TRMG nodes/edges/history (Section 6), reusing the
exact same `TemporalReasoningMemoryGraph` API every other CMI component
reads from (`graph/trmg/model.py`): telemetry and communications arrive as
`ingest_telemetry_edge` calls, trust evolution is `NodeHistory.trust_trajectory()`,
software/firmware/sensor state is TRMG node attributes and confidence.
`MissionDigitalTwin` (`twin.py`) does not duplicate any of that into a
second, parallel state model -- see this module's docstring for why.

What genuinely has no TRMG-node analogue -- mission phase, which
objectives have completed, the operator-action log, the cyber-incident
log, and whether the mission has been aborted/completed -- lives in
`MissionEnvironment` below instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.outcome import MissionOutcome

MISSION_PHASES = (
    "pre_flight",
    "taxi",
    "takeoff",
    "cruise",
    "approach",
    "landing",
    "complete",
    "aborted",
)


@dataclass
class MissionEnvironment:
    """The twin-level state Section 15 implies but that the TRMG has no
    node for: mission phase, objective/operator/cyber-incident logs, and
    terminal status."""

    mission_id: str
    time: float = 0.0
    phase: str = "pre_flight"
    objectives_complete: set[str] = field(default_factory=set)
    operator_actions: list[tuple[float, str]] = field(default_factory=list)
    cyber_incident_log: list[tuple[float, str]] = field(default_factory=list)
    aborted: bool = False
    completed: bool = False
    outcome: MissionOutcome | None = None
    """The Digital Twin's ground-truth outcome for this mission run
    (Section 15: "the twin is now also the source of the ground-truth
    outcome records o that populate Episodic Memory"). Set by a
    `MissionComplete` event, not inferred."""

    def set_phase(self, phase: str, time: float) -> None:
        if phase not in MISSION_PHASES:
            raise ValueError(f"unknown mission phase {phase!r}; must be one of {MISSION_PHASES}")
        self.phase = phase
        self.time = time

    def record_operator_action(self, time: float, action: str) -> None:
        self.operator_actions.append((time, action))

    def record_cyber_incident(self, time: float, description: str) -> None:
        self.cyber_incident_log.append((time, description))
