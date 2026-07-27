"""Wires the Digital Twin (Section 15) into Algorithm 1 (`TTRNPipeline`,
Section 5.2) as the producer/consumer relationship this phase requires:
"The Digital Twin becomes the producer. Existing AI modules become
consumers." (Phase 7 brief).

`run_scenario_with_pipeline` never builds a second graph: it schedules a
`ScenarioSpec`'s events onto `twin` and drives `pipeline.infer` directly
against `twin.graph`, the same `TemporalReasoningMemoryGraph` instance
every Digital Twin event mutates (`digital_twin/events.py`), deriving the
per-node feature vectors Section 7.1's attention embedding needs from
`twin.node_feature_vector` (seeded, deterministic -- Section 15's replay
requirement). The full downstream chain (TRMG -> Memory -> Cyber
Reasoning -> Argumentation -> Causal Engine -> Scheduler ->
Explainability) is exactly Algorithm 1's own control flow inside
`TTRNPipeline.infer`; this module adds nothing to that chain, it only
supplies the chain's first input from simulated rather than handcrafted
mission state.

Each `TwinQueryPoint` also carries a `MissionDependencyRiskIndex` (Section
12.6.2, `cyber.metrics.mission_dependency_risk_index`) snapshot computed
against `twin.graph` at that exact query time, with an empty
`intrinsic_risk` mapping: this repository has no vulnerability database
(see `cyber/README.md`), so every dependency node's own vulnerability risk
rho(v) is honestly 0 rather than fabricated -- MDRI's structural term
(dependency centrality over the TRMG) is still real and non-zero. MTES and
CMCS (Sections 12.6.1/12.6.3) are not computed here because both
additionally require a specific vulnerability node, an integration window,
and (for CMCS) P_success -- scenario-specific inputs this generic
integration point has no non-fabricated way to supply for an arbitrary
scenario; see `cyber/scheduler_integration.py` for where MTES is computed
with real, scenario-supplied vulnerability context.

The MDRI snapshot -- and the mission-state fields (`mission_phase`,
`mission_completed`, `mission_outcome`, `cyber_incident_count`) -- are
captured inside the query loop immediately after `twin.run(until=t)`, not
recomputed afterward from the fully-run twin: `twin.graph`'s edge
confidences and `twin.env`'s fields keep changing as later events apply,
so any snapshot taken after the full scenario finishes would silently
describe the twin's *final* state at every query time instead of its state
at `t`. `digital_twin/dataset.py` reads these fields directly rather than
re-deriving them, for exactly this reason.

Because `MissionDigitalTwin.__init__` always constructs its own fresh
`TemporalReasoningMemoryGraph`, the only way for a `TTRNPipeline` to share
that exact graph instance is for the twin to be constructed first and the
pipeline second (`TTRNPipeline(graph=twin.graph, ...)`); this module does
not construct either for the caller, so that ordering stays explicit at
the call site instead of being hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyber.metrics import mission_dependency_risk_index
from digital_twin.scenarios import ScenarioSpec
from digital_twin.twin import MissionDigitalTwin
from memory.outcome import MissionOutcome
from reasoning.ttrn.pipeline import TTRNPipeline, TTRNResult


@dataclass(frozen=True)
class TwinQueryPoint:
    """One `TTRNResult`, produced by querying `pipeline` against the twin's
    state at simulation time `time`, together with a same-instant snapshot
    of twin-level mission state and the MDRI cyber metric (see module
    docstring for why these are captured here rather than after the fact)."""

    time: float
    result: TTRNResult
    mission_phase: str
    mission_completed: bool
    mission_outcome: MissionOutcome | None
    cyber_incident_count: int
    mdri_value: float
    mdri_dependency_node_count: int


def run_scenario_with_pipeline(
    twin: MissionDigitalTwin,
    spec: ScenarioSpec,
    pipeline: TTRNPipeline,
    *,
    feature_dim: int = 8,
) -> tuple[TwinQueryPoint, ...]:
    """Schedule every event in `spec` onto `twin`, then advance `twin` to
    each of `spec.query_times` (ascending order) and invoke
    `pipeline.infer` against the twin's own graph at that point, producing
    one `TTRNResult` per query time.

    `pipeline` must already be constructed with `graph=twin.graph` (the
    same instance): this function does not construct the pipeline itself,
    so callers keep full control over which memory / argumentation /
    causal / scheduler collaborators are wired in, per the
    dependency-injection pattern established in `reasoning/ttrn/pipeline.py`.

    At each query time, `outcome` is passed to `pipeline.infer` only once
    the twin has actually completed (`twin.env.completed`) -- i.e. only the
    query time at or after a `MissionComplete` event carries a ground-truth
    outcome, matching Algorithm 1 line 15's `PendingOutcome(m)` semantics
    (an in-progress mission has no outcome yet to record).

    Node features for the attention embedding (Section 7.1) are drawn from
    `spec.mission_node`'s k-hop neighborhood (`pipeline.k_hop`) plus
    `spec.mission_node` itself, using `twin.node_feature_vector` --
    deterministic per (twin.seed, node_id), so repeated calls with the same
    seed and scenario reproduce identical `TTRNResult`s (Section 15's
    replay requirement, `digital_twin/replay.py`).
    """
    if pipeline.graph is not twin.graph:
        raise ValueError(
            "pipeline must be constructed with graph=twin.graph (the same "
            "instance) so the Digital Twin remains the single producer of "
            "mission state; got a different TemporalReasoningMemoryGraph"
        )

    for event in spec.events:
        twin.schedule(event)

    mission_node = spec.mission_node
    results: list[TwinQueryPoint] = []
    for t in sorted(spec.query_times):
        twin.run(until=t)

        neighborhood = twin.graph.k_hop_neighborhood(mission_node, pipeline.k_hop)
        node_ids = neighborhood | {mission_node}
        node_features = {
            node_id: twin.node_feature_vector(node_id, dim=feature_dim) for node_id in node_ids
        }

        outcome = twin.env.outcome if twin.env.completed else None
        result = pipeline.infer(
            mission_node,
            t,
            node_features,
            outcome=outcome,
        )

        mdri = mission_dependency_risk_index(
            twin.graph, mission_node, intrinsic_risk={}, k_hop=pipeline.k_hop
        )
        results.append(
            TwinQueryPoint(
                time=t,
                result=result,
                mission_phase=twin.env.phase,
                mission_completed=twin.env.completed,
                mission_outcome=twin.env.outcome,
                cyber_incident_count=len(twin.env.cyber_incident_log),
                mdri_value=mdri.value,
                mdri_dependency_node_count=len(mdri.dependency_nodes),
            )
        )

    return tuple(results)
