"""The Digital Twin's event catalog (manuscript Section 15).

Every event mutates the twin's real `TemporalReasoningMemoryGraph` and/or
`MissionEnvironment` -- never a separate world model -- reusing existing
TRMG APIs exactly as any other telemetry source would (Algorithm 1, line
1: "ingest new telemetry edges"). Four events (`FirmwareUpdate`,
`PatchCompleted`) reuse `cyber.risk_propagation.refresh_dependency_trust`
(Section 12.3) directly rather than reimplementing dependency-trust
semantics a second time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cyber.risk_propagation import refresh_dependency_trust
from memory.outcome import MissionOutcome

if TYPE_CHECKING:
    from digital_twin.twin import MissionDigitalTwin


@dataclass(frozen=True)
class SimulationEvent(ABC):
    """Base class for every event the Digital Twin's discrete-event engine
    can schedule and apply."""

    time: float

    @abstractmethod
    def apply(self, twin: MissionDigitalTwin) -> None:
        """Mutate `twin.graph` and/or `twin.env` to reflect this event
        having occurred at `self.time`."""
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


def _ensure_node(twin: MissionDigitalTwin, node_id: str, **attributes: object) -> None:
    if not twin.graph.has_node(node_id):
        twin.graph.add_node(node_id, **attributes)
    elif attributes:
        twin.graph.get_node(node_id).attributes.update(attributes)


@dataclass(frozen=True)
class MissionStart(SimulationEvent):
    """The mission begins (Section 15): creates the mission node if
    absent and sets the initial mission phase."""

    mission_node: str = ""
    phase: str = "pre_flight"

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.mission_node)
        twin.env.set_phase(self.phase, self.time)


@dataclass(frozen=True)
class WaypointReached(SimulationEvent):
    """A mission objective/waypoint is completed."""

    waypoint_id: str = ""

    def apply(self, twin: MissionDigitalTwin) -> None:
        twin.env.objectives_complete.add(self.waypoint_id)
        twin.env.record_operator_action(self.time, f"waypoint reached: {self.waypoint_id}")


@dataclass(frozen=True)
class CommunicationLoss(SimulationEvent):
    """A communication link degrades: Section 12.5's "communication-link
    trust is treated as a TRMG node subject to the same confidence-decay
    mechanism as any other evidence" -- modeled here as new, degraded
    telemetry for the comm_node -> mission_node edge (Algorithm 1, line 1)."""

    comm_node: str = ""
    mission_node: str = ""
    residual_confidence: float = 0.05
    decay_rate: float = 0.05
    priority: float = 0.7

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.comm_node)
        _ensure_node(twin, self.mission_node)
        twin.graph.ingest_telemetry_edge(
            self.comm_node,
            self.mission_node,
            timestamp=self.time,
            confidence=self.residual_confidence,
            decay_rate=self.decay_rate,
            priority=self.priority,
        )


@dataclass(frozen=True)
class SensorFailure(SimulationEvent):
    """A sensor fails or degrades."""

    sensor_node: str = ""
    mission_node: str = ""
    residual_confidence: float = 0.1
    decay_rate: float = 0.05
    priority: float = 0.6

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.sensor_node, operational=False)
        _ensure_node(twin, self.mission_node)
        twin.graph.ingest_telemetry_edge(
            self.sensor_node,
            self.mission_node,
            timestamp=self.time,
            confidence=self.residual_confidence,
            decay_rate=self.decay_rate,
            priority=self.priority,
        )


@dataclass(frozen=True)
class WeatherChange(SimulationEvent):
    """Weather conditions change, per Section 8.2.3's `weather_category`
    feature and Section 12.5's decay-subject comm/evidence trust."""

    weather_node: str = ""
    mission_node: str = ""
    condition: str = "clear"
    severity: float = 0.0
    """0.0 (no impact) to 1.0 (maximal impact on downstream confidence)."""
    decay_rate: float = 0.03
    priority: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0,1], got {self.severity}")

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.weather_node, weather_category=self.condition)
        _ensure_node(twin, self.mission_node)
        twin.graph.ingest_telemetry_edge(
            self.weather_node,
            self.mission_node,
            timestamp=self.time,
            confidence=max(0.0, 1.0 - self.severity),
            decay_rate=self.decay_rate,
            priority=self.priority,
        )


@dataclass(frozen=True)
class FirmwareUpdate(SimulationEvent):
    """Firmware is updated to a new version: Section 12.3's "firmware
    trust specifically is reasoned over via secure-boot and attestation
    state" -- reuses `refresh_dependency_trust` directly rather than a
    second trust-update mechanism."""

    firmware_node: str = ""
    new_version: str = ""
    new_trust: float = 1.0
    refresh_event: str = "attestation"

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.firmware_node, dependency_identity=self.new_version)
        node = twin.graph.get_node(self.firmware_node)
        node.attributes["dependency_identity"] = self.new_version
        refresh_dependency_trust(node, self.time, self.new_trust, self.refresh_event)


@dataclass(frozen=True)
class SoftwareCompromise(SimulationEvent):
    """A software or firmware component is compromised (Section 12.2's
    attack-chain entry point / Section 12.1's originating vulnerability
    node): degrades the component's own trust history and logs a cyber
    incident, rather than a `refresh_dependency_trust` event (compromise
    is not a refresh -- trust moves the opposite direction)."""

    software_node: str = ""
    mission_node: str = ""
    exploit: str = ""
    residual_trust: float = 0.1
    entry_node_type: str | None = None

    def apply(self, twin: MissionDigitalTwin) -> None:
        attrs: dict[str, object] = {}
        if self.entry_node_type is not None:
            attrs["entry_node_type"] = self.entry_node_type
        _ensure_node(twin, self.software_node, **attrs)
        node = twin.graph.get_node(self.software_node)
        node.history.append_trust(self.time, self.residual_trust)
        node.history.append_reasoning(
            self.time,
            f"compromised via {self.exploit}",
            source="digital_twin.SoftwareCompromise",
        )
        _ensure_node(twin, self.mission_node)
        if not twin.graph.has_edge(self.software_node, self.mission_node):
            twin.graph.ingest_telemetry_edge(
                self.software_node,
                self.mission_node,
                timestamp=self.time,
                confidence=self.residual_trust,
                decay_rate=0.08,
                priority=0.8,
            )
        twin.env.record_cyber_incident(
            self.time, f"{self.software_node} compromised via {self.exploit}"
        )


@dataclass(frozen=True)
class PatchCompleted(SimulationEvent):
    """A previously-compromised or vulnerable component is patched:
    reuses `refresh_dependency_trust` (Section 12.3) directly -- the same
    mechanism `FirmwareUpdate` uses -- since a completed patch is exactly
    a "patch_application" refresh event."""

    node_id: str = ""
    new_trust: float = 1.0

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.node_id)
        node = twin.graph.get_node(self.node_id)
        refresh_dependency_trust(node, self.time, self.new_trust, "patch_application")


@dataclass(frozen=True)
class ThreatDetected(SimulationEvent):
    """A threat, anomaly, or overdue maintenance condition is detected and
    logged -- deliberately generic (free-text `description`) so it also
    covers conditions Section 15.1's scenario list names without a
    dedicated event type of their own (e.g. overdue maintenance)."""

    node_id: str = ""
    description: str = ""

    def apply(self, twin: MissionDigitalTwin) -> None:
        _ensure_node(twin, self.node_id)
        node = twin.graph.get_node(self.node_id)
        node.history.append_reasoning(
            self.time, f"threat detected: {self.description}", source="digital_twin.ThreatDetected"
        )
        twin.env.record_cyber_incident(self.time, self.description)


@dataclass(frozen=True)
class MissionAbort(SimulationEvent):
    """The mission is aborted."""

    reason: str = ""

    def apply(self, twin: MissionDigitalTwin) -> None:
        twin.env.aborted = True
        twin.env.set_phase("aborted", self.time)
        twin.env.record_operator_action(self.time, f"mission aborted: {self.reason}")


@dataclass(frozen=True)
class MissionResume(SimulationEvent):
    """A previously aborted mission resumes."""

    resume_phase: str = "cruise"
    reason: str = ""

    def apply(self, twin: MissionDigitalTwin) -> None:
        twin.env.aborted = False
        twin.env.set_phase(self.resume_phase, self.time)
        note = f"mission resumed: {self.reason}" if self.reason else "mission resumed"
        twin.env.record_operator_action(self.time, note)


@dataclass(frozen=True)
class MissionComplete(SimulationEvent):
    """The mission concludes with a ground-truth outcome (Section 15:
    "the twin is now also the source of the ground-truth outcome records
    o that populate Episodic Memory," Section 8.2.3)."""

    outcome: MissionOutcome = MissionOutcome.SUCCESS

    def apply(self, twin: MissionDigitalTwin) -> None:
        twin.env.completed = True
        twin.env.outcome = self.outcome
        twin.env.set_phase("complete", self.time)


ALL_EVENT_TYPES: tuple[type[SimulationEvent], ...] = (
    MissionStart,
    WaypointReached,
    CommunicationLoss,
    SensorFailure,
    WeatherChange,
    FirmwareUpdate,
    SoftwareCompromise,
    PatchCompleted,
    ThreatDetected,
    MissionAbort,
    MissionResume,
    MissionComplete,
)
