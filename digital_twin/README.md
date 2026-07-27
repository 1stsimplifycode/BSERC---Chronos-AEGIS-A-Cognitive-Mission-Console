# digital_twin/ -- Mission Digital Twin (Section 15)

**Status: implemented.** A real discrete-event simulation engine, not a
visualization and not a game: every state transition advances simulation
time, and every event mutates the same `TemporalReasoningMemoryGraph`
(Section 6) the rest of the framework already reads from.

## Component map

| Module | Manuscript tie-in | What it is |
|---|---|---|
| `clock.py` | Section 15's "temporal simulation" | `SimulationClock`: monotonic simulation time, cannot move backwards |
| `queue.py` | discrete-event scheduling | `EventQueue`: a min-heap ordered by `(event.time, insertion_sequence)` -- insertion order is the deterministic tiebreak for same-time events |
| `events.py` | Section 15's twelve named event types | `SimulationEvent` (ABC) + `MissionStart`, `WaypointReached`, `CommunicationLoss`, `SensorFailure`, `WeatherChange`, `FirmwareUpdate`, `SoftwareCompromise`, `PatchCompleted`, `ThreatDetected`, `MissionAbort`, `MissionResume`, `MissionComplete` |
| `state.py` | twin-level state with no TRMG-node analogue | `MissionEnvironment`: phase, completed objectives, operator-action log, cyber-incident log, terminal status |
| `twin.py` | the engine itself | `MissionDigitalTwin`: owns one real TRMG + one `MissionEnvironment`, applies scheduled events in time order |
| `scenarios.py` | Section 15.1's scenario library | `ScenarioSpec` + the ten canonical `build_*` scenarios, a `normal_mission` baseline, two composite scenarios, and (Phase 8) five firmware/MDRI-exercising scenarios |
| `replay.py` | "allowing experiments to replay identical scenarios" | `run_scenario` / `replay`: verifies a scenario reproduces an identical event trajectory and node-feature fingerprints for the same seed |
| `integration.py` | "the Digital Twin becomes the producer" | `run_scenario_with_pipeline`: schedules a scenario onto a twin and drives `TTRNPipeline.infer` (Algorithm 1) against the twin's own graph at each query time |
| `dataset.py` | "This becomes the reproducibility dataset" | `MissionRecord` + CSV/JSON export, flattened from `TwinQueryPoint`s |
| `visualization.py` | mission timeline / trust evolution / metric dashboards | matplotlib-based plots over computed `MissionRecord` data (optional dependency, see module docstring) |

## Design decisions worth knowing before reading the code

**No duplicated state.** Section 15 lists aircraft state, telemetry,
communications, software, firmware, sensors, weather, cyber incidents, and
trust evolution as things the twin simulates. All of those already have a
home as TRMG nodes/edges/history (Section 6) -- `MissionDigitalTwin` does
not build a second, parallel world model for any of them. `state.py`'s
`MissionEnvironment` holds *only* the handful of things that genuinely have
no TRMG-node analogue: mission phase, which objectives are complete, the
operator-action log, the cyber-incident log, and terminal status
(aborted/completed/outcome).

**Events mutate the twin, never AI modules directly.** Every
`SimulationEvent.apply(twin)` writes to `twin.graph` and/or `twin.env`
using the TRMG's own existing API (`ingest_telemetry_edge`,
`NodeHistory.append_trust`, `cyber.risk_propagation.refresh_dependency_trust`)
-- the same calls any other telemetry source would make. Nothing in
`events.py` reaches into `reasoning/`, `memory/`, `agents/`, `cyber/`, or
`scheduler/`.

**The twin is a producer, the pipeline is a consumer.** `twin.py` never
imports from `reasoning/`, `memory/`, `agents/`, `cyber/`, or `scheduler/`.
`integration.py` is the one place that hands the twin's real graph to
`TTRNPipeline` (Section 5's Algorithm 1) -- unchanged, exactly as it would
be handed a hand-built graph in `examples/`. From there, Algorithm 1's own
control flow (TRMG -> Memory -> Argumentation -> Causal Engine -> Scheduler
-> Explainability) is untouched by this package.

**Twelve event types, not one per scenario.** Section 15.1 names ten
scenarios but the twelve event types are the actual primitive vocabulary.
Two scenarios (`sensor_spoofing`, `maintenance_delay`/`runway_closure`/
`supply_chain_delay`) have no dedicated event type of their own and are
built as documented compositions of the closest existing event(s) --
`SensorFailure` + `ThreatDetected` for spoofing, `ThreatDetected` +
`MissionAbort`/`MissionResume` for the three delay-shaped scenarios -- see
each `build_*` function's own docstring for the specific reasoning.

**Deterministic by construction -- across processes, not just within
one.** `MissionDigitalTwin.node_feature_vector` derives a feature vector
from `hashlib.sha256(f"{seed}:{node_id}")`, and every `SimulationEvent` is
an immutable, fully-specified dataclass scheduled up front from a
`ScenarioSpec`. `replay.py` verifies this empirically (same seed ->
identical applied-event sequence, identical feature fingerprints) rather
than just asserting it. **Phase 8 fix:** this method originally used
Python's built-in `hash((seed, node_id))`, which is salted per *process*
by `PYTHONHASHSEED` (on by default since Python 3.3) -- the same seed and
node id produced a *different* feature vector in every separate process
invocation, while looking perfectly deterministic within any single
process (which is why no test caught it until Phase 8 hashed
`scripts/reproduce_all.py`'s output across two separate `python`
invocations). `hashlib.sha256` is not affected by `PYTHONHASHSEED`. See
`docs/reproducibility-guide.md` for the full account and
`tests/scripts/test_reproduce_all.py`'s subprocess-based regression test.

## Scenario library: canonical vs. firmware (Phase 8)

`cyber.centrality.dependency_nodes` (Section 12.6.2's Dep(m)) identifies
dependency nodes via the `dependency_identity` node attribute, and of the
event types only `FirmwareUpdate` ever sets it. None of Section 15.1's
ten canonical scenarios (`CANONICAL_SCENARIO_BUILDERS`, left completely
unmodified since Phase 7) schedule a `FirmwareUpdate` event, so `Dep(m)`
is provably empty for all of them -- discovered, not assumed, by running
`scripts/experiment_rq12_cyber_metrics.py`. Phase 8 added
`FIRMWARE_SCENARIO_BUILDERS` (`build_firmware_update_in_flight`,
`build_delayed_firmware_rollout`, `build_compromised_firmware_deployment`,
`build_staged_patch_rollout`, `build_mixed_firmware_and_communication_degradation`)
specifically to close this gap: each schedules a `FirmwareUpdate` event
paired with a second, edge-creating event on the same node (no single
existing event type both sets `dependency_identity` and creates an edge
to the mission node -- see `scenarios.py`'s own Phase 8 docstring for why
this composition, not a new event type, was used), giving every one of
them a genuinely non-empty `Dep(m) = {that node}`. `mdri_value` itself
(as opposed to `|Dep(m)|`) still remains `0.0` for every scenario in both
groups -- that requires a vulnerability database this repository does not
have, a separate, still-open limitation adding scenarios cannot fix.

## Usage

```python
from digital_twin.scenarios import build_satellite_blackout
from digital_twin.twin import MissionDigitalTwin
from digital_twin.integration import run_scenario_with_pipeline
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline

spec = build_satellite_blackout()
twin = MissionDigitalTwin(spec.name, seed=0)
pipeline = TTRNPipeline(twin.graph, MissionAwareGNN(feature_dim=8, seed=0), k_hop=6)

query_points = run_scenario_with_pipeline(twin, spec, pipeline)
# one TwinQueryPoint (a real TTRNResult + a same-instant mission-state
# snapshot) per spec.query_times entry
```

See `digital_twin/dataset.py` to flatten `query_points` into a
`MissionRecord` dataset, `evaluation/batch.py` to sweep this over many
scenarios/seeds, and `scripts/experiment_rq12_cyber_metrics.py` /
`scripts/experiment_rq6_rq7_memory_retrieval.py` for complete, runnable
examples.

## Tests

`tests/digital_twin/` covers every module above (clock, queue, events,
state, twin, scenarios, replay, integration, dataset) at 100% line
coverage, including the five new firmware scenarios and the
`hashlib`-based determinism fix; `visualization.py` is covered up to the
point matplotlib would be required (an optional dependency not installed
in this environment -- see its module docstring).
