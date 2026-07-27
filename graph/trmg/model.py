"""The Temporal Trust Graph (TTG) and its extension to the Temporal Reasoning
Memory Graph (TRMG).

Faithful to manuscript Sections 6.1-6.2:

Section 6.1 (TTG, retained): every edge e carries a fixed attribute tuple
    chi(e) = (t, w, c, r, lambda, d, pi)
where t is the timestamp of the edge's last update, w is its validity window,
c is a confidence value, r is a risk contribution, lambda is a decay rate
governing how c degrades with elapsed time since t, d is a dependency
indicator linking the edge to other edges it structurally depends on, and pi
is a priority weight used when multiple edges compete for a node's attention
in downstream reasoning.

Section 6.2 (TRMG extension): every node n additionally carries a history
tuple
    rho(n) = (rho_reason(n), rho_trust(n), rho_agent(n), rho_cf(n),
              rho_unc(n), rho_ev(n))
TRMG = (TTG, rho): the same graph structure as the TTG, with every node
carrying a persistent, queryable reasoning history. All six logs are
append-only (mirroring the append-only discipline Proposition 4, Section 5.4,
requires of Episodic Memory): nothing in this module ever mutates or removes
a past history entry; only Section 8's consolidation (not yet implemented,
see memory/interfaces.py) is permitted to derive summaries from it.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from graph.trmg.snapshot import TRMGSnapshot


class ReasoningLogEntry(NamedTuple):
    """One entry of rho_reason(n): a past reasoning conclusion attached to n."""

    timestamp: float
    conclusion: str
    source: str  # which pipeline stage / algorithm produced this conclusion


class TrustLogEntry(NamedTuple):
    """One entry of rho_trust(n): a past mission-trust value T(n, t')."""

    timestamp: float
    trust: float


class AgentLogEntry(NamedTuple):
    """One entry of rho_agent(n): which agent (Section 9.1) touched n and
    whether its hypothesis was accepted or attacked in argumentation."""

    timestamp: float
    agent_id: str
    accepted: bool


class CounterfactualLogEntry(NamedTuple):
    """One entry of rho_cf(n): a past counterfactual query (Section 10.5)
    evaluated at or through n."""

    timestamp: float
    query: str
    result_summary: str


class UncertaintyLogEntry(NamedTuple):
    """One entry of rho_unc(n): a past uncertainty estimate sigma at n."""

    timestamp: float
    sigma: float


class EvidenceLogEntry(NamedTuple):
    """One entry of rho_ev(n): an evidence object (Section 9.1's
    AssembleEvidence output) that has previously cited n."""

    timestamp: float
    evidence_id: str
    description: str


@dataclass
class NodeHistory:
    """rho(n): the six append-only logs the TRMG attaches to every node.

    Each `append_*` method is the only way to add an entry; there is
    deliberately no `update_*` or `remove_*` method, so the append-only
    discipline of Proposition 4 (Section 5.4) is enforced by the type, not
    by convention.
    """

    reason: list[ReasoningLogEntry] = field(default_factory=list)
    trust: list[TrustLogEntry] = field(default_factory=list)
    agent: list[AgentLogEntry] = field(default_factory=list)
    counterfactual: list[CounterfactualLogEntry] = field(default_factory=list)
    uncertainty: list[UncertaintyLogEntry] = field(default_factory=list)
    evidence: list[EvidenceLogEntry] = field(default_factory=list)

    def append_reasoning(self, timestamp: float, conclusion: str, source: str) -> None:
        self.reason.append(ReasoningLogEntry(timestamp, conclusion, source))

    def append_trust(self, timestamp: float, trust: float) -> None:
        self.trust.append(TrustLogEntry(timestamp, trust))

    def append_agent(self, timestamp: float, agent_id: str, accepted: bool) -> None:
        self.agent.append(AgentLogEntry(timestamp, agent_id, accepted))

    def append_counterfactual(self, timestamp: float, query: str, result_summary: str) -> None:
        self.counterfactual.append(CounterfactualLogEntry(timestamp, query, result_summary))

    def append_uncertainty(self, timestamp: float, sigma: float) -> None:
        self.uncertainty.append(UncertaintyLogEntry(timestamp, sigma))

    def append_evidence(self, timestamp: float, evidence_id: str, description: str) -> None:
        self.evidence.append(EvidenceLogEntry(timestamp, evidence_id, description))

    def trust_trajectory(self) -> list[tuple[float, float]]:
        """xi(n) = {(t_i, c_i) : t_i in rho_trust(n)} (Section 6.5)."""
        return [(e.timestamp, e.trust) for e in self.trust]

    def prune(self, keep: int) -> int:
        """Temporal pruning (Phase 8's TRMG Evolution requirement; Section
        6.4 names "how aggressively to prune or summarize old entries" as
        an open empirical question this method answers with the same
        oldest-first-by-time policy `memory.episodic.EpisodicMemoryStore.prune_oldest`
        already uses for Episodic Memory, applied per log rather than
        arbitrarily).

        Keeps the `keep` most recent entries of *each* of the six logs
        (each log already stores entries in append order, i.e. oldest
        first, so "most recent" is simply the last `keep` elements).
        Returns the total number of entries removed across all six logs.

        This is the one operation in this module that is not itself
        reversible -- call `TemporalReasoningMemoryGraph.snapshot()`
        before pruning if the pre-prune state must remain recoverable
        (Phase 8: "every update must remain reversible for replay").
        """
        removed = 0
        for log in (
            self.reason,
            self.trust,
            self.agent,
            self.counterfactual,
            self.uncertainty,
            self.evidence,
        ):
            if len(log) > keep:
                removed += len(log) - keep
                del log[: len(log) - keep]
        return removed


@dataclass
class TRMGNode:
    """A TRMG node: an identifier, domain-specific attributes, and rho(n)."""

    node_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    history: NodeHistory = field(default_factory=NodeHistory)


@dataclass
class EdgeAttributes:
    """chi(e) = (t, w, c, r, lambda, d, pi) (Section 6.1)."""

    timestamp: float
    validity_window: float
    confidence: float
    risk: float
    decay_rate: float
    dependency: frozenset[str] = field(default_factory=frozenset)
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be in [0,1], got {self.priority}")
        if self.decay_rate < 0.0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")

    def decayed_confidence(self, now: float) -> float:
        """c_u(t) * e^{-lambda_u * (t - t_u)}: confidence at `now`, decayed
        since this edge's last update (Section 6.1, Section 7.2)."""
        import math

        elapsed = max(0.0, now - self.timestamp)
        return self.confidence * math.exp(-self.decay_rate * elapsed)


class TemporalReasoningMemoryGraph:
    """TRMG = (TTG, rho): a typed property graph of TRMGNodes connected by
    EdgeAttributes-carrying directed edges, with append-only reasoning
    history attached to every node.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TRMGNode] = {}
        # adjacency: source -> {target -> EdgeAttributes}
        self._out_edges: dict[str, dict[str, EdgeAttributes]] = {}
        self._in_edges: dict[str, dict[str, EdgeAttributes]] = {}

    # -- node/edge construction -------------------------------------------------

    def add_node(self, node_id: str, **attributes: Any) -> TRMGNode:
        if node_id in self._nodes:
            raise ValueError(f"node '{node_id}' already exists")
        node = TRMGNode(node_id=node_id, attributes=dict(attributes))
        self._nodes[node_id] = node
        self._out_edges[node_id] = {}
        self._in_edges[node_id] = {}
        return node

    def get_node(self, node_id: str) -> TRMGNode:
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        timestamp: float,
        confidence: float,
        risk: float = 0.0,
        decay_rate: float = 0.05,
        validity_window: float = float("inf"),
        priority: float = 1.0,
        dependency: Iterable[str] = (),
    ) -> EdgeAttributes:
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("both source and target nodes must exist before adding an edge")
        attrs = EdgeAttributes(
            timestamp=timestamp,
            validity_window=validity_window,
            confidence=confidence,
            risk=risk,
            decay_rate=decay_rate,
            dependency=frozenset(dependency),
            priority=priority,
        )
        self._out_edges[source][target] = attrs
        self._in_edges[target][source] = attrs
        return attrs

    def get_edge(self, source: str, target: str) -> EdgeAttributes:
        return self._out_edges[source][target]

    def has_edge(self, source: str, target: str) -> bool:
        return target in self._out_edges.get(source, {})

    def out_edges(self, node_id: str) -> dict[str, EdgeAttributes]:
        return dict(self._out_edges.get(node_id, {}))

    def in_edges(self, node_id: str) -> dict[str, EdgeAttributes]:
        return dict(self._in_edges.get(node_id, {}))

    @property
    def nodes(self) -> Iterator[str]:
        return iter(self._nodes.keys())

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._out_edges.values())

    # -- Phase 8 TRMG Evolution: snapshots, restore, temporal pruning ------------

    def snapshot(self, label: str) -> TRMGSnapshot:
        """Capture a deep, point-in-time copy of this graph's full state
        (see graph/trmg/snapshot.py). Does not mutate this graph."""
        from graph.trmg.snapshot import TRMGSnapshot

        return TRMGSnapshot(
            label=label,
            nodes=copy.deepcopy(self._nodes),
            out_edges=copy.deepcopy(self._out_edges),
            in_edges=copy.deepcopy(self._in_edges),
        )

    def restore(self, snapshot: TRMGSnapshot) -> None:
        """Replace this graph's live state with a deep copy of `snapshot`
        (deep-copied again on the way in, so the snapshot itself stays
        untouched and can be restored from more than once). This is what
        makes prior operations -- including temporal pruning
        (`NodeHistory.prune`), which is not itself reversible -- reversible
        for replay: restore whatever snapshot was taken before them.
        """
        self._nodes = copy.deepcopy(snapshot.nodes)
        self._out_edges = copy.deepcopy(snapshot.out_edges)
        self._in_edges = copy.deepcopy(snapshot.in_edges)

    def prune_history(self, keep: int) -> int:
        """Apply `NodeHistory.prune(keep)` to every node in this graph.
        Returns the total number of history entries removed. See
        `NodeHistory.prune` for the retention policy and its
        (documented, non-reversible-in-place) semantics.
        """
        return sum(node.history.prune(keep) for node in self._nodes.values())

    # -- Algorithm 1, line 1: decay + telemetry ingestion ------------------------

    def apply_decay(self, now: float) -> None:
        """Algorithm 1, line 1 (first half): 'Apply decay lambda to all edge
        confidences in G_t since the last update.'

        Decay is not destructive rewriting of history: this method rewrites
        each edge's stored confidence to its decayed value at `now` and
        advances its timestamp, so that subsequent decay calls compound
        correctly from the new reference time. It does not touch any
        NodeHistory log.
        """
        for source, targets in self._out_edges.items():
            for target, attrs in targets.items():
                decayed = attrs.decayed_confidence(now)
                new_attrs = EdgeAttributes(
                    timestamp=now,
                    validity_window=attrs.validity_window,
                    confidence=decayed,
                    risk=attrs.risk,
                    decay_rate=attrs.decay_rate,
                    dependency=attrs.dependency,
                    priority=attrs.priority,
                )
                self._out_edges[source][target] = new_attrs
                self._in_edges[target][source] = new_attrs

    def ingest_telemetry_edge(
        self,
        source: str,
        target: str,
        *,
        timestamp: float,
        confidence: float,
        risk: float = 0.0,
        decay_rate: float = 0.05,
        validity_window: float = float("inf"),
        priority: float = 1.0,
        dependency: Iterable[str] = (),
    ) -> EdgeAttributes:
        """Algorithm 1, line 1 (second half): 'ingest new telemetry edges.'
        Creates the endpoint nodes if they do not already exist, then adds
        (or overwrites) the edge with fresh telemetry-derived attributes.
        """
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
        return self.add_edge(
            source,
            target,
            timestamp=timestamp,
            confidence=confidence,
            risk=risk,
            decay_rate=decay_rate,
            validity_window=validity_window,
            priority=priority,
            dependency=dependency,
        )

    # -- Algorithm 1, line 2: k-hop neighborhood ----------------------------------

    def k_hop_neighborhood(self, node_id: str, k: int) -> set[str]:
        """N_k(m): the k-hop neighborhood of `node_id`, traversing edges in
        either direction (a node's trust can depend on both its predecessors
        and successors). Bounded to k <= 6 per Section 5.1's design target;
        this method accepts any k but callers should respect that bound.
        """
        if node_id not in self._nodes:
            raise KeyError(node_id)
        visited = {node_id}
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= k:
                continue
            neighbors = set(self._out_edges.get(current, {})) | set(self._in_edges.get(current, {}))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        return visited
