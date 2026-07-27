"""Trust Propagation (manuscript Section 7.2).

Trust Propagation is the mechanism by which a trust-relevant change at one
TRMG node is transmitted to every other node whose trust estimate depends on
it, before any of those posteriors are individually updated -- a
deterministic, graph-structural computation, kept explicitly separate from
the probabilistic Bayesian update of Section 7.3.

    rho(u->v, t) = c_u(t) * e^{-lambda_u * (t - t_u)} * pi_(u,v)
    T_in(v, t)   = max_{u : (u,v) in E} rho(u->v, t)

i.e. u's confidence at its last update, exponentially decayed at rate
lambda_u over the elapsed time since t_u, weighted by the priority of the
(u,v) edge; v's total propagated trust input is the priority-weighted
maximum over its incoming edges.

Proposition 1 (Section 5.4): T_in(v,t) is computed in at most k <= 6 steps
and is well-defined independent of cycles beyond that bound, because rho is
a closed-form function of u's own attributes, not a fixed-point iteration.

Proposition 2 (Section 5.4): if every c_u(t) in [0,1], every decay factor in
(0,1], and every pi in [0,1], then rho(u->v,t) in [0,1] and T_in(v,t) in
[0,1] -- a product/max of terms in [0,1] cannot leave [0,1]. `EdgeAttributes`
already enforces confidence and priority in [0,1] at construction time
(graph/trmg/model.py), so this module does not re-validate those bounds.
"""

from __future__ import annotations

from graph.trmg.model import EdgeAttributes, TemporalReasoningMemoryGraph


def propagated_trust_contribution(edge: EdgeAttributes, now: float) -> float:
    """rho(u->v, t) for a single incoming edge (Section 7.2)."""
    return edge.decayed_confidence(now) * edge.priority


def total_propagated_trust_input(
    graph: TemporalReasoningMemoryGraph, node_id: str, now: float
) -> float:
    """T_in(v, t) = max over v's incoming edges of rho(u->v, t).

    Returns 0.0 for a node with no incoming edges: an isolated node has no
    propagated trust input to report, which is the honest value (not an
    error, and not a fabricated default like 1.0).
    """
    incoming = graph.in_edges(node_id)
    if not incoming:
        return 0.0
    return max(propagated_trust_contribution(edge, now) for edge in incoming.values())
