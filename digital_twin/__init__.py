"""The Mission Digital Twin (manuscript Section 15): a discrete-event
simulation engine that generates the TRMG (Section 6) every other CMI
component reads from -- the shared substrate on which the framework's
research questions (Section 17) are, in principle, evaluated.

digital_twin.clock          SimulationClock: monotonic simulation time
digital_twin.queue          EventQueue: discrete-event scheduling
digital_twin.state          MissionEnvironment: twin-level state with no
                             TRMG-node analogue (phase, objectives, logs)
digital_twin.events         SimulationEvent + the 12 named event types
digital_twin.twin           MissionDigitalTwin: the engine itself
digital_twin.scenarios      Section 15.1's ten named scenarios
digital_twin.replay         Deterministic replay support
digital_twin.integration    Wiring the twin's TRMG into TTRNPipeline
digital_twin.dataset        CSV/JSON dataset export
"""

from digital_twin.clock import SimulationClock
from digital_twin.dataset import (
    MissionRecord,
    build_dataset,
    build_record,
    to_json,
    write_csv,
    write_json,
)
from digital_twin.events import (
    ALL_EVENT_TYPES,
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
from digital_twin.integration import TwinQueryPoint, run_scenario_with_pipeline
from digital_twin.queue import EventQueue
from digital_twin.replay import ReplayLog, ReplayMismatchError, replay, run_scenario
from digital_twin.scenarios import (
    ALL_SCENARIO_BUILDERS,
    CANONICAL_SCENARIO_BUILDERS,
    ScenarioSpec,
)
from digital_twin.state import MISSION_PHASES, MissionEnvironment
from digital_twin.twin import MissionDigitalTwin
from digital_twin.visualization import (
    plot_metric_dashboard,
    plot_mission_timeline,
    plot_scenario_comparison,
    plot_scheduler_decisions,
    plot_trust_evolution,
)

__all__ = [
    "SimulationClock",
    "ALL_EVENT_TYPES",
    "CommunicationLoss",
    "FirmwareUpdate",
    "MissionAbort",
    "MissionComplete",
    "MissionResume",
    "MissionStart",
    "PatchCompleted",
    "SensorFailure",
    "SimulationEvent",
    "SoftwareCompromise",
    "ThreatDetected",
    "WaypointReached",
    "WeatherChange",
    "EventQueue",
    "MISSION_PHASES",
    "MissionEnvironment",
    "MissionDigitalTwin",
    "TwinQueryPoint",
    "run_scenario_with_pipeline",
    "ReplayLog",
    "ReplayMismatchError",
    "replay",
    "run_scenario",
    "ALL_SCENARIO_BUILDERS",
    "CANONICAL_SCENARIO_BUILDERS",
    "ScenarioSpec",
    "MissionRecord",
    "build_dataset",
    "build_record",
    "to_json",
    "write_csv",
    "write_json",
    "plot_mission_timeline",
    "plot_metric_dashboard",
    "plot_scenario_comparison",
    "plot_scheduler_decisions",
    "plot_trust_evolution",
]
