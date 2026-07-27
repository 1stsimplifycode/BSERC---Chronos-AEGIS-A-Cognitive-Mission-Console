"""Mission-Aware Cyber Risk Propagation (manuscript Section 12), base
mechanism (12.1), supply-chain/firmware trust (12.3), and zero-trust
reasoning (12.4).

12.1 (retained): "a vulnerability is treated as a risk contribution
originating at a software or firmware TRMG node and propagated,
attenuated at each hop, along dependency edges to whatever downstream
nodes it can reach -- a navigation module, a flight computer, a
satellite-communication link, and ultimately the mission and
operational-trust nodes themselves -- feeding directly into the cyber
impact head C(m,t) of Section 7."

ASSUMPTION (documented): Section 12.1 does not give a closed-form
per-hop attenuation formula of its own. It explicitly says the propagated
risk "feed[s] directly into the cyber impact head C(m,t) of Section 7,"
i.e. the same graph the mission-aware GNN and Trust Propagation (Section
7.2) already operate over. This module therefore reuses Section 7.2's
exact per-edge attenuation factor,

    rho(u->v, t) = c_u(t) * e^{-lambda_u*(t-t_u)} * pi_(u,v)

(reasoning/trust/propagation.py, unchanged) and extends it across
multiple hops as the *widest path* (maximum-product path) from the
originating vulnerability node to each reachable node -- the natural
multi-hop generalization of Section 7.2's own single-hop
"priority-weighted maximum over incoming edges" rule, rather than an
unrelated attenuation formula invented for this module. This is what
Section 12.6.1's `attenuation(v->mission)` term (see cyber/metrics.py)
is computed from.
"""

from __future__ import annotations

import heapq
import math

from graph.trmg.model import TemporalReasoningMemoryGraph, TRMGNode
from reasoning.trust.propagation import propagated_trust_contribution, total_propagated_trust_input


def propagate_risk_attenuation(
    graph: TemporalReasoningMemoryGraph, source_node: str, now: float, k_hop: int = 6
) -> dict[str, float]:
    """attenuation(source -> v, t) for every v within `k_hop` of
    `source_node`: the maximum-product-path attenuation factor,
    generalizing rho(u->v,t) (Section 7.2) across multiple hops (Section
    12.1's "propagated, attenuated at each hop... to whatever downstream
    nodes it can reach").

    Returns a dict including `source_node` itself (attenuation 1.0 -- no
    hops traversed yet). A node not present in the result was not reached
    within `k_hop` hops; treat its attenuation as 0.0 (see
    `attenuation_to`), which is the honest value for "risk from this
    vulnerability does not structurally reach that node," not an error.

    Implemented as a Dijkstra-style widest-path search (max-heap via
    negation): since every per-edge factor is in [0,1] and hop
    contributions compose multiplicatively (a strictly monotonic,
    order-preserving combination), the same greedy-relaxation argument
    that makes Dijkstra correct for additive shortest paths makes this
    correct for multiplicative widest paths.
    """
    return {node: att for node, (att, _decay) in _propagate(graph, source_node, now, k_hop).items()}


def propagate_risk_attenuation_with_decay(
    graph: TemporalReasoningMemoryGraph, source_node: str, now: float, k_hop: int = 6
) -> dict[str, tuple[float, float]]:
    """As `propagate_risk_attenuation`, but each node also carries the sum
    of `decay_rate` over the edges of its winning (widest) path from
    `source_node`. This aggregate decay rate is what
    `cyber.metrics.mission_trust_exposure_score` (Section 12.6.1) uses to
    project attenuation forward in time for its integral: since
    attenuation is a product of per-edge exp(-lambda_i*(tau-t_i)) decay
    terms, attenuation(source->v, tau) = attenuation(source->v, t) *
    exp(-Lambda*(tau-t)) where Lambda is exactly this path-summed decay
    rate -- an exact identity given the winning path stays the same
    across the integration window (documented assumption, see
    cyber/metrics.py).
    """
    return _propagate(graph, source_node, now, k_hop)


def _propagate(
    graph: TemporalReasoningMemoryGraph, source_node: str, now: float, k_hop: int
) -> dict[str, tuple[float, float]]:
    """Shared widest-path search backing both `propagate_risk_attenuation`
    and `propagate_risk_attenuation_with_decay`."""
    if not graph.has_node(source_node):
        raise KeyError(source_node)

    neighborhood = graph.k_hop_neighborhood(source_node, k_hop)
    best: dict[str, tuple[float, float]] = {source_node: (1.0, 0.0)}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(-1.0, source_node)]

    while heap:
        neg_att, node = heapq.heappop(heap)
        att = -neg_att
        if node in visited:
            continue
        visited.add(node)
        _cur_att, cur_decay = best[node]
        for target, edge in graph.out_edges(node).items():
            if target not in neighborhood:
                continue
            candidate_att = att * propagated_trust_contribution(edge, now)
            candidate_decay = cur_decay + edge.decay_rate
            if candidate_att > best.get(target, (0.0, 0.0))[0]:
                best[target] = (candidate_att, candidate_decay)
                heapq.heappush(heap, (-candidate_att, target))

    return best


def attenuation_to(
    graph: TemporalReasoningMemoryGraph,
    source_node: str,
    target_node: str,
    now: float,
    k_hop: int = 6,
) -> float:
    """attenuation(source_node -> target_node, t): 0.0 if `target_node` is
    not reachable within `k_hop` hops (an honest "no structural path,"
    Section 12.1's own scoping via Section 5.1's k <= 6 bound)."""
    return propagate_risk_attenuation(graph, source_node, now, k_hop).get(target_node, 0.0)


# -- Section 12.3: Mission-Aware Software Supply-Chain and Firmware Trust --

REFRESH_EVENTS = ("patch_application", "attestation", "secure_boot", "clean_audit")
"""Section 12.3: dependency trust "is refreshed by patch application,
attestation events, or a clean audit"; firmware trust specifically "via
secure-boot and attestation state.\""""


def dependency_node_trust(node: TRMGNode, now: float, decay_rate: float = 0.05) -> float:
    """A software/firmware dependency node's current trust: its most
    recent rho_trust(n) entry (Section 6.2), decayed forward to `now` by
    the same TTG decay mechanism (Section 6.1) any edge confidence uses --
    "each software or firmware dependency node in the TRMG carries a trust
    value that decays according to the same decay mechanism lambda as any
    other TTG edge... rather than being treated as a static property fixed
    at the dependency's introduction" (Section 12.3).

    Returns 0.0 if the node has no trust history yet -- honestly "never
    established," not a fabricated default.
    """
    trajectory = node.history.trust_trajectory()
    if not trajectory:
        return 0.0
    last_time, last_trust = trajectory[-1]
    elapsed = max(0.0, now - last_time)
    return last_trust * math.exp(-decay_rate * elapsed)


def refresh_dependency_trust(node: TRMGNode, now: float, new_trust: float, event: str) -> None:
    """Refresh a dependency node's trust following `event` (Section
    12.3): appends to the node's own history (graph/trmg/model.py's
    `NodeHistory`, the same append-only mechanism every other TRMG
    reasoning trace uses) rather than a separate cyber-specific log.
    """
    if event not in REFRESH_EVENTS:
        raise ValueError(f"event must be one of {REFRESH_EVENTS}, got {event!r}")
    if not 0.0 <= new_trust <= 1.0:
        raise ValueError(f"new_trust must be in [0,1], got {new_trust}")
    node.history.append_trust(now, new_trust)
    node.history.append_reasoning(
        now,
        f"dependency trust refreshed to {new_trust:.4f} via {event}",
        source="cyber.risk_propagation.refresh_dependency_trust",
    )


# -- Section 12.4: Zero-Trust Reasoning --------------------------------------


def verify_identity_trust(
    graph: TemporalReasoningMemoryGraph, identity_node: str, now: float
) -> float:
    """Section 12.4: "every credential or identity node's trust value is
    continuously re-derived from current evidence rather than granted
    persistently once verified." This function holds no cache to
    invalidate -- it always recomputes Trust Propagation (Section 7.2)
    fresh from the current graph state, which *is* the zero-trust
    property, not merely a description of it.
    """
    return total_propagated_trust_input(graph, identity_node, now)


def scoped_neighborhood_for_agent(
    graph: TemporalReasoningMemoryGraph, mission_node: str, k_hop: int = 6
) -> frozenset[str]:
    """Section 12.4: "every agent's access to a TRMG subgraph is scoped to
    the minimum neighborhood required for its hypothesis-generation role
    (Section 9.1)." Since Section 9.1's nine concrete domain roles are not
    yet implemented (agents/README.md), this scopes every agent uniformly
    to the mission's own k-hop neighborhood -- least-privilege relative to
    the *whole* TRMG, though not yet role-differentiated within that
    neighborhood; a documented limitation, not silently ignored scope.
    """
    return frozenset(graph.k_hop_neighborhood(mission_node, k_hop))
