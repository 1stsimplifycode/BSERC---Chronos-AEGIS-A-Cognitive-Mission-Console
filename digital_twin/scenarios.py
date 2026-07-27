"""The Scenario Library (manuscript Section 15.1, retained).

"We retain, unchanged, the scenario library specified in our preceding
formulation: satellite blackout, GPS degradation, adverse weather,
maintenance delay, runway closure, software exploit injection, sensor
spoofing, communication loss, fuel shortage, and supply-chain delay, each
implemented as a defined intervention on specific TRMG nodes with known
ground truth."

Each `build_*` function below returns a `ScenarioSpec`: a fixed sequence
of `digital_twin.events.SimulationEvent`s (the "defined intervention on
specific TRMG nodes") together with a `ground_truth_outcome` -- the twin's
own design-time label for what this scenario represents (Section 15:
"the twin is now also the source of the ground-truth outcome records o").
This is a simulation-design label, not a claimed evaluation result: no
scenario here reports that CMI *achieved* this outcome, only that this is
what the scenario was constructed to represent, exactly as Section 15.1
specifies ("with known ground truth").

`NORMAL_MISSION` is a baseline with no injected intervention, needed as
the control case for comparison experiments (`evaluation/`) but not itself
one of Section 15.1's ten named scenarios. `MULTIPLE_SIMULTANEOUS_FAILURES`
and `MIXED_CYBER_OPERATIONAL` are composites built by combining two of the
ten canonical scenarios' own events -- they introduce no new intervention
mechanism, only combine faithfully-implemented ones, matching this
phase's request for such composite scenarios without overstating Section
15.1's own (exactly ten) canonical list.

PHASE 8 ADDITION -- `FIRMWARE_SCENARIO_BUILDERS`: `CANONICAL_SCENARIO_BUILDERS`
(the original ten, above) is left completely unmodified, preserved exactly
as Phase 7 built it for before/after comparison. Running
`scripts/experiment_rq12_cyber_metrics.py` against those ten scenarios
found MDRI's structural term (`|Dep(m)|`, Section 12.6.2) is provably zero
for every one of them, because none schedules a `FirmwareUpdate` event --
the only one of the twelve event types (`digital_twin/events.py`) that
sets the `dependency_identity` node attribute `cyber.centrality.dependency_nodes`
keys on. The five scenarios below exist specifically to exercise that gap
with genuine dependency nodes: firmware update in flight, delayed firmware
rollout, compromised firmware deployment, staged patch rollout, and mixed
firmware/communication degradation.

None of the twelve event types both sets `dependency_identity` *and*
creates an edge to the mission node (`FirmwareUpdate` only touches the
firmware node itself, per its own docstring) -- so, exactly like
`sensor_spoofing`'s and `maintenance_delay`'s existing documented
compositions elsewhere in this module, every scenario below pairs
`FirmwareUpdate` with a second, edge-creating event on the *same* node id:
`CommunicationLoss` (reused purely for its `ingest_telemetry_edge` side
effect -- to connect the firmware node to the mission graph via the
channel that would carry a rollout, not necessarily a degraded one) for
the benign/delayed cases, and `SoftwareCompromise` (which both creates the
edge and models trust degradation) for the compromised case. This is the
closest faithful composition available from the existing twelve event
types, documented here rather than left as a silent approximation, and it
introduces no new event type or architectural module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from digital_twin.events import (
    CommunicationLoss,
    FirmwareUpdate,
    MissionAbort,
    MissionComplete,
    MissionResume,
    MissionStart,
    PatchCompleted,
    SensorFailure,
    SimulationEvent,
    SoftwareCompromise,
    ThreatDetected,
    WaypointReached,
    WeatherChange,
)
from memory.outcome import MissionOutcome

DEFAULT_MISSION_NODE = "mission"


@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario: a fixed event sequence with a known (simulation
    design-time) ground truth outcome (Section 15.1)."""

    name: str
    description: str
    mission_node: str
    events: tuple[SimulationEvent, ...]
    ground_truth_outcome: MissionOutcome
    intervention_nodes: tuple[str, ...]
    query_times: tuple[float, ...] = field(default_factory=tuple)
    """Suggested simulation times at which to query the reasoning
    pipeline (`digital_twin.integration.run_scenario_with_pipeline`)."""


def build_normal_mission(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """Baseline: no injected intervention (not one of Section 15.1's ten;
    the control case comparison experiments need)."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="taxi"),
        WaypointReached(time=2.0, waypoint_id="departure"),
        WaypointReached(time=6.0, waypoint_id="cruise_checkpoint"),
        MissionComplete(time=10.0, outcome=MissionOutcome.SUCCESS),
    )
    return ScenarioSpec(
        name="normal_mission",
        description="No injected intervention; the mission proceeds as planned.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.SUCCESS,
        intervention_nodes=(),
        query_times=(0.0, 5.0, 10.0),
    )


def build_satellite_blackout(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        CommunicationLoss(
            time=2.0, comm_node="satcom", mission_node=mission_node, residual_confidence=0.02
        ),
        ThreatDetected(time=2.5, node_id="satcom", description="satellite blackout detected"),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="satellite_blackout",
        description="Total loss of satellite communication mid-mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("satcom",),
        query_times=(0.0, 2.0, 5.0, 10.0),
    )


def build_gps_degradation(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        SensorFailure(
            time=2.0, sensor_node="gps", mission_node=mission_node, residual_confidence=0.15
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="gps_degradation",
        description="GPS navigation signal degrades to near-unusable accuracy.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("gps",),
        query_times=(0.0, 2.0, 10.0),
    )


def build_adverse_weather(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        WeatherChange(
            time=2.0,
            weather_node="weather",
            mission_node=mission_node,
            condition="storm",
            severity=0.7,
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="adverse_weather",
        description="A storm system develops along the mission's route.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("weather",),
        query_times=(0.0, 2.0, 10.0),
    )


def build_maintenance_delay(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="pre_flight"),
        ThreatDetected(time=1.0, node_id="engine", description="maintenance overdue"),
        MissionAbort(time=2.0, reason="maintenance required before departure"),
        MissionResume(time=4.0, resume_phase="taxi", reason="maintenance completed"),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="maintenance_delay",
        description="Overdue maintenance is detected before departure, delaying the mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("engine",),
        query_times=(0.0, 1.0, 4.0, 10.0),
    )


def build_runway_closure(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="taxi"),
        ThreatDetected(time=1.0, node_id="runway", description="runway closure notified"),
        MissionAbort(time=1.5, reason="destination runway closed"),
        MissionResume(time=5.0, resume_phase="taxi", reason="alternate runway cleared"),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="runway_closure",
        description="The destination runway closes, forcing a hold and diversion.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("runway",),
        query_times=(0.0, 1.5, 5.0, 10.0),
    )


def build_software_exploit_injection(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        SoftwareCompromise(
            time=2.0,
            software_node="flight_software",
            mission_node=mission_node,
            exploit="CVE-2024-1111",
            residual_trust=0.05,
            entry_node_type="internet_facing_gateway",
        ),
        ThreatDetected(
            time=2.5, node_id="flight_software", description="exploit signature detected"
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.CYBER_INCIDENT),
    )
    return ScenarioSpec(
        name="software_exploit_injection",
        description="A known CVE is exploited against a flight-software component mid-mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.CYBER_INCIDENT,
        intervention_nodes=("flight_software",),
        query_times=(0.0, 2.0, 10.0),
    )


def build_sensor_spoofing(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """Modeled via `SensorFailure` (degraded confidence) plus a
    `ThreatDetected` marking the failure as suspected spoofing rather than
    an ordinary hardware fault -- the event catalog does not distinguish
    "sensor reports plausible-looking wrong data" from "sensor stops
    reporting," so this is the closest faithful composition of the twelve
    named event types (documented, not a silent approximation)."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        SensorFailure(
            time=2.0, sensor_node="imu", mission_node=mission_node, residual_confidence=0.3
        ),
        ThreatDetected(
            time=2.2,
            node_id="imu",
            description="sensor spoofing suspected: conflicting cross-checks",
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.CYBER_INCIDENT),
    )
    return ScenarioSpec(
        name="sensor_spoofing",
        description="An inertial measurement sensor reports conflicting, suspected-spoofed data.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.CYBER_INCIDENT,
        intervention_nodes=("imu",),
        query_times=(0.0, 2.0, 10.0),
    )


def build_communication_loss(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """Distinct from `satellite_blackout`: a local radio link degrades
    (partial, not total, loss), matching Section 15.1 listing both
    scenarios separately."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        CommunicationLoss(
            time=3.0, comm_node="vhf_radio", mission_node=mission_node, residual_confidence=0.1
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="communication_loss",
        description="A local VHF radio link degrades mid-mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("vhf_radio",),
        query_times=(0.0, 3.0, 10.0),
    )


def build_fuel_shortage(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        ThreatDetected(
            time=3.0, node_id="fuel_system", description="fuel reserves below threshold"
        ),
        MissionAbort(time=3.5, reason="insufficient fuel to complete mission"),
        MissionComplete(time=10.0, outcome=MissionOutcome.FAILURE),
    )
    return ScenarioSpec(
        name="fuel_shortage",
        description="Fuel reserves drop below the safe threshold mid-mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.FAILURE,
        intervention_nodes=("fuel_system",),
        query_times=(0.0, 3.0, 3.5, 10.0),
    )


def build_supply_chain_delay(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="pre_flight"),
        ThreatDetected(
            time=1.0,
            node_id="avionics_component",
            description="supply-chain delay: component unavailable",
        ),
        MissionAbort(time=1.5, reason="required component not delivered"),
        MissionResume(time=6.0, resume_phase="taxi", reason="component delivered"),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="supply_chain_delay",
        description="A required avionics component is delayed in the supply chain before takeoff.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("avionics_component",),
        query_times=(0.0, 1.5, 6.0, 10.0),
    )


CANONICAL_SCENARIO_BUILDERS = (
    build_satellite_blackout,
    build_gps_degradation,
    build_adverse_weather,
    build_maintenance_delay,
    build_runway_closure,
    build_software_exploit_injection,
    build_sensor_spoofing,
    build_communication_loss,
    build_fuel_shortage,
    build_supply_chain_delay,
)
"""Section 15.1's exactly-ten named scenarios, in the order the manuscript
lists them."""


def build_multiple_simultaneous_failures(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """Composite: combines `gps_degradation` and `adverse_weather`'s own
    events (offset in time to interleave), not a new intervention
    mechanism -- two of Section 15.1's ten canonical scenarios co-occurring."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        SensorFailure(
            time=2.0, sensor_node="gps", mission_node=mission_node, residual_confidence=0.15
        ),
        WeatherChange(
            time=2.5,
            weather_node="weather",
            mission_node=mission_node,
            condition="storm",
            severity=0.6,
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="multiple_simultaneous_failures",
        description="GPS degradation and adverse weather occur within the same mission window.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("gps", "weather"),
        query_times=(0.0, 2.0, 2.5, 10.0),
    )


def build_mixed_cyber_operational(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """Composite: combines `software_exploit_injection` and
    `communication_loss`'s own events -- a cyber failure and an
    operational failure co-occurring."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        SoftwareCompromise(
            time=2.0,
            software_node="flight_software",
            mission_node=mission_node,
            exploit="CVE-2024-1111",
            residual_trust=0.05,
            entry_node_type="internet_facing_gateway",
        ),
        CommunicationLoss(
            time=2.8, comm_node="vhf_radio", mission_node=mission_node, residual_confidence=0.1
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.CYBER_INCIDENT),
    )
    return ScenarioSpec(
        name="mixed_cyber_operational",
        description="A software exploit and an independent communication-link failure co-occur.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.CYBER_INCIDENT,
        intervention_nodes=("flight_software", "vhf_radio"),
        query_times=(0.0, 2.0, 2.8, 10.0),
    )


def build_firmware_update_in_flight(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """A routine firmware update is pushed and applied successfully
    mid-mission. `CommunicationLoss` is reused here purely for its
    edge-creation side effect (see module docstring's Phase 8 note): the
    high residual confidence reflects a healthy rollout channel, not a
    degraded one."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        CommunicationLoss(
            time=2.0,
            comm_node="avionics_firmware",
            mission_node=mission_node,
            residual_confidence=0.9,
            decay_rate=0.02,
            priority=0.6,
        ),
        FirmwareUpdate(
            time=2.5,
            firmware_node="avionics_firmware",
            new_version="avionics_fw@4.1",
            new_trust=1.0,
            refresh_event="patch_application",
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.SUCCESS),
    )
    return ScenarioSpec(
        name="firmware_update_in_flight",
        description="A routine firmware update is pushed and applied successfully mid-mission.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.SUCCESS,
        intervention_nodes=("avionics_firmware",),
        query_times=(0.0, 2.0, 2.5, 10.0),
    )


def build_delayed_firmware_rollout(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """A firmware rollout misses its staging window before departure,
    delaying the mission until it completes late."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="pre_flight"),
        CommunicationLoss(
            time=1.0,
            comm_node="nav_firmware",
            mission_node=mission_node,
            residual_confidence=0.6,
            decay_rate=0.02,
            priority=0.5,
        ),
        ThreatDetected(
            time=1.2, node_id="nav_firmware", description="firmware rollout delayed: staging"
            " window missed"
        ),
        MissionAbort(time=1.5, reason="firmware rollout incomplete before departure"),
        FirmwareUpdate(
            time=5.0,
            firmware_node="nav_firmware",
            new_version="nav_fw@2.3",
            new_trust=1.0,
            refresh_event="patch_application",
        ),
        MissionResume(time=5.5, resume_phase="taxi", reason="rollout completed late"),
        MissionComplete(time=10.0, outcome=MissionOutcome.DELAY),
    )
    return ScenarioSpec(
        name="delayed_firmware_rollout",
        description="A firmware rollout misses its pre-departure staging window and completes"
        " late.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.DELAY,
        intervention_nodes=("nav_firmware",),
        query_times=(0.0, 1.5, 5.5, 10.0),
    )


def build_compromised_firmware_deployment(
    mission_node: str = DEFAULT_MISSION_NODE,
) -> ScenarioSpec:
    """A tampered firmware version is deployed and its compromise is
    subsequently detected. `SoftwareCompromise` supplies the edge to the
    mission node (see module docstring's Phase 8 note) as well as the
    trust degradation."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        FirmwareUpdate(
            time=2.0,
            firmware_node="satcom_firmware",
            new_version="satcom_fw@3.0-tampered",
            new_trust=0.3,
            refresh_event="attestation",
        ),
        SoftwareCompromise(
            time=2.2,
            software_node="satcom_firmware",
            mission_node=mission_node,
            exploit="supply-chain firmware tampering",
            residual_trust=0.05,
            entry_node_type="supply_chain",
        ),
        ThreatDetected(
            time=2.5,
            node_id="satcom_firmware",
            description="compromised firmware deployment detected",
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.CYBER_INCIDENT),
    )
    return ScenarioSpec(
        name="compromised_firmware_deployment",
        description="A tampered firmware version is deployed and its compromise is detected"
        " in flight.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.CYBER_INCIDENT,
        intervention_nodes=("satcom_firmware",),
        query_times=(0.0, 2.0, 2.5, 10.0),
    )


def build_staged_patch_rollout(mission_node: str = DEFAULT_MISSION_NODE) -> ScenarioSpec:
    """A flight-control firmware patch rolls out in three stages, each
    raising trust further via `PatchCompleted`'s reuse of
    `refresh_dependency_trust` (Section 12.3) -- the same mechanism
    `patch_completed` events use elsewhere in this module, applied
    repeatedly to model progressive rollout."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        CommunicationLoss(
            time=1.0,
            comm_node="flight_control_firmware",
            mission_node=mission_node,
            residual_confidence=0.7,
            decay_rate=0.02,
            priority=0.6,
        ),
        FirmwareUpdate(
            time=1.5,
            firmware_node="flight_control_firmware",
            new_version="fcs_fw@1.9-stage1",
            new_trust=0.7,
            refresh_event="patch_application",
        ),
        PatchCompleted(time=4.0, node_id="flight_control_firmware", new_trust=0.9),
        PatchCompleted(time=7.0, node_id="flight_control_firmware", new_trust=1.0),
        MissionComplete(time=10.0, outcome=MissionOutcome.SUCCESS),
    )
    return ScenarioSpec(
        name="staged_patch_rollout",
        description="A flight-control firmware patch rolls out in three progressive stages.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.SUCCESS,
        intervention_nodes=("flight_control_firmware",),
        query_times=(0.0, 1.5, 4.0, 7.0, 10.0),
    )


def build_mixed_firmware_and_communication_degradation(
    mission_node: str = DEFAULT_MISSION_NODE,
) -> ScenarioSpec:
    """A firmware update with uncertain attestation co-occurs with an
    independent communication-link degradation on a different node --
    Phase 8's requested "mixed firmware and communication degradation"
    scenario, composed the same way `mixed_cyber_operational` composes two
    independent failures."""
    events = (
        MissionStart(time=0.0, mission_node=mission_node, phase="cruise"),
        FirmwareUpdate(
            time=2.0,
            firmware_node="satcom_firmware",
            new_version="satcom_fw@2.0",
            new_trust=0.6,
            refresh_event="attestation",
        ),
        CommunicationLoss(
            time=2.1,
            comm_node="satcom_firmware",
            mission_node=mission_node,
            residual_confidence=0.6,
            decay_rate=0.02,
            priority=0.6,
        ),
        CommunicationLoss(
            time=3.0,
            comm_node="vhf_radio",
            mission_node=mission_node,
            residual_confidence=0.15,
            decay_rate=0.05,
        ),
        MissionComplete(time=10.0, outcome=MissionOutcome.CYBER_INCIDENT),
    )
    return ScenarioSpec(
        name="mixed_firmware_and_communication_degradation",
        description="An uncertain firmware attestation co-occurs with an independent radio-link"
        " degradation.",
        mission_node=mission_node,
        events=events,
        ground_truth_outcome=MissionOutcome.CYBER_INCIDENT,
        intervention_nodes=("satcom_firmware", "vhf_radio"),
        query_times=(0.0, 2.0, 3.0, 10.0),
    )


FIRMWARE_SCENARIO_BUILDERS = (
    build_firmware_update_in_flight,
    build_delayed_firmware_rollout,
    build_compromised_firmware_deployment,
    build_staged_patch_rollout,
    build_mixed_firmware_and_communication_degradation,
)
"""Phase 8's five firmware/MDRI-exercising scenarios (see module
docstring). `CANONICAL_SCENARIO_BUILDERS` above remains exactly Section
15.1's original ten, unmodified, for before/after comparison."""


ALL_SCENARIO_BUILDERS = (
    build_normal_mission,
    *CANONICAL_SCENARIO_BUILDERS,
    build_multiple_simultaneous_failures,
    build_mixed_cyber_operational,
    *FIRMWARE_SCENARIO_BUILDERS,
)
