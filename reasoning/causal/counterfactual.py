"""The Counterfactual Engine (manuscript Section 10.1, retained).

"Given a candidate intervention do(x -> x') on a specific TRMG node or edge
attribute, the engine recomputes the mission-aware GNN's forward pass over
the intervened graph and reports T(m, x->x') = T(m | do(x->x')) - T(m | x),
together with any secondary affected quantity."

SCOPE: the concrete "TRMG node or edge attribute" this module intervenes on
is an edge's confidence value -- the one edge attribute the mission-aware
GNN's attention mechanism (Section 7.1, reasoning/trust/gnn.py) actually
consumes as a weighting input. The worked examples of Section 10.3.1
(patch_status, R_comm, navigation_confidence, ...) name mission-specific
variables this repository's TRMG schema does not yet model (they belong to
Sections 11-14, not yet implemented); intervening on those would require
inventing mission semantics the manuscript does not specify at the code
level, so this module is scoped to the one intervention target the current
TRMG + GNN can support faithfully. Algorithm 5 (Section 10.5) uses this as
its correlational fallback branch (line 6-7) regardless of which specific
variable a candidate intervention names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from reasoning.trust.gnn import MissionAwareGNN


@dataclass(frozen=True)
class EdgeInterventionRequest:
    """do(x -> x'): set edge (source, target)'s confidence to `value`."""

    source: str
    target: str
    value: float


def _clone_with_intervention(
    graph: TemporalReasoningMemoryGraph, intervention: EdgeInterventionRequest, now: float
) -> TemporalReasoningMemoryGraph:
    """A copy of `graph` with `intervention`'s edge confidence replaced,
    leaving every other node/edge unchanged -- the graph-mutilation analog
    of do() (reasoning/causal/scm.py) at the TRMG level rather than the SCM
    level.
    """
    clone = TemporalReasoningMemoryGraph()
    for node_id in graph.nodes:
        node = graph.get_node(node_id)
        clone.add_node(node_id, **node.attributes)

    for source in graph.nodes:
        for target, edge in graph.out_edges(source).items():
            confidence = (
                intervention.value
                if (source, target) == (intervention.source, intervention.target)
                else edge.decayed_confidence(now)
            )
            clone.add_edge(
                source,
                target,
                timestamp=now,
                confidence=confidence,
                risk=edge.risk,
                decay_rate=edge.decay_rate,
                validity_window=edge.validity_window,
                priority=edge.priority,
                dependency=edge.dependency,
            )
    return clone


def correlational_counterfactual(
    graph: TemporalReasoningMemoryGraph,
    gnn: MissionAwareGNN,
    mission_node: str,
    node_features: dict[str, np.ndarray],
    now: float,
    intervention: EdgeInterventionRequest,
) -> float:
    """T(m, x->x') = T(m | do(x->x')) - T(m | x) (Section 10.1): re-run the
    GNN's forward pass over an intervened copy of the graph and compare
    against the un-intervened baseline.
    """
    if not graph.has_edge(intervention.source, intervention.target):
        raise KeyError(
            f"cannot intervene on edge ({intervention.source!r}, {intervention.target!r}): "
            f"no such edge exists in the graph"
        )

    baseline_embedding = gnn.embed(graph, mission_node, node_features, now)
    baseline_trust = gnn.predict(baseline_embedding).trust

    intervened_graph = _clone_with_intervention(graph, intervention, now)
    intervened_embedding = gnn.embed(intervened_graph, mission_node, node_features, now)
    intervened_trust = gnn.predict(intervened_embedding).trust

    return intervened_trust - baseline_trust
