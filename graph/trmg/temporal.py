"""Temporal Dependency Analysis (manuscript Section 6.5).

Formalizes the explicit tracking and querying of how a node's dependency
structure itself changes over time, as distinct from how a node's confidence
or trust changes while its dependency structure stays fixed.

    Dep(n, t)              the set of nodes n depends on at time t
    DeltaDep(n, t)         Dep(n, t) XOR Dep(n, t - delta_t)   (symmetric difference)
    Vol(n, t, w)           |union of DeltaDep(n, t_i) for t-w <= t_i <= t| / |Dep(n, t)|

Vol is the fraction of n's current dependency set that has changed at least
once within a trailing window w; Section 12 (cyber risk propagation) and
Section 9 (argumentation) both treat an unusually volatile dependency
structure as evidence of elevated risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.trmg.model import TemporalReasoningMemoryGraph


@dataclass(frozen=True)
class DependencySnapshot:
    """A recorded Dep(n, t): the dependency set of a node at a point in time."""

    timestamp: float
    dependencies: frozenset[str]


class TemporalDependencyAnalyzer:
    """Maintains a per-node history of DependencySnapshots and computes
    dependency evolution and volatility over a TRMG.

    This is a distinct analytical procedure over the TRMG (Section 6.5), not
    a byproduct of the base TRMG model: the base graph only stores the
    *current* dependency structure implicit in its edges; this class is what
    makes Dep(n, t) queryable across time.
    """

    def __init__(self, graph: TemporalReasoningMemoryGraph) -> None:
        self._graph = graph
        self._snapshots: dict[str, list[DependencySnapshot]] = {}

    def record_snapshot(self, node_id: str, timestamp: float) -> DependencySnapshot:
        """Capture Dep(node_id, timestamp) from the graph's current incoming
        edges (a node's dependencies are the sources of its in-edges) and
        append it to that node's snapshot history. Callers are expected to
        call this whenever the graph's edges touching `node_id` change,
        e.g. immediately after `TemporalReasoningMemoryGraph.add_edge` or
        `ingest_telemetry_edge`.
        """
        deps = frozenset(self._graph.in_edges(node_id).keys())
        snapshot = DependencySnapshot(timestamp=timestamp, dependencies=deps)
        self._snapshots.setdefault(node_id, []).append(snapshot)
        return snapshot

    def dependencies_at(self, node_id: str, t: float) -> frozenset[str]:
        """Dep(n, t): the most recent recorded snapshot at or before t."""
        history = self._snapshots.get(node_id, [])
        candidates = [s for s in history if s.timestamp <= t]
        if not candidates:
            return frozenset()
        return max(candidates, key=lambda s: s.timestamp).dependencies

    def dependency_delta(self, node_id: str, t: float, previous_t: float) -> frozenset[str]:
        """DeltaDep(n, t) = Dep(n, t) XOR Dep(n, t - delta_t)."""
        return self.dependencies_at(node_id, t) ^ self.dependencies_at(node_id, previous_t)

    def volatility(self, node_id: str, t: float, window: float) -> float:
        """Vol(n, t, w): fraction of n's current dependency set that changed
        at least once within the trailing window [t - w, t].

        Returns 0.0 if Dep(n, t) is empty (nothing to normalize against),
        which is the honest degenerate case rather than a division error.
        """
        current = self.dependencies_at(node_id, t)
        if not current:
            return 0.0

        history = sorted(
            (s for s in self._snapshots.get(node_id, []) if t - window <= s.timestamp <= t),
            key=lambda s: s.timestamp,
        )
        changed: set[str] = set()
        for prev, nxt in zip(history, history[1:], strict=False):
            changed |= prev.dependencies ^ nxt.dependencies
        return len(changed & current) / len(current) if changed else 0.0

    def confidence_trajectory(self, node_id: str) -> list[tuple[float, float]]:
        """xi(n): the node's confidence/trust trajectory, read directly from
        rho_trust(n) (Section 6.5's definition; the trajectory itself lives
        on the node's NodeHistory, not on this analyzer)."""
        return self._graph.get_node(node_id).history.trust_trajectory()
