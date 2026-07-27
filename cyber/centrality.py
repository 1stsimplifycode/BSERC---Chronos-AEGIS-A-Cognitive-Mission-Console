"""Structural centrality over the TRMG's dependency subgraph, for Section
12.6.2's Mission Dependency Risk Index.

Section 12.6.2: "MDRI(m) = sum_{v in Dep(m)} rho(v)*centrality(v,TRMG),
where Dep(m) is the set of software/firmware dependency nodes reachable
from mission m, and centrality(v,TRMG) is v's structural centrality within
the dependency subgraph (for example, betweenness centrality restricted to
dependency edges)."

This module implements the specific realization the manuscript names as
its example: betweenness centrality (Brandes, 2001), computed over the
subgraph induced by Dep(m) and the TRMG edges between its members ("the
dependency subgraph... restricted to dependency edges").
"""

from __future__ import annotations

from collections import deque

from graph.trmg.model import TemporalReasoningMemoryGraph


def dependency_nodes(
    graph: TemporalReasoningMemoryGraph, mission_node: str, k_hop: int = 6
) -> frozenset[str]:
    """Dep(m): the set of software/firmware dependency nodes reachable
    from mission m (Section 12.6.2). Identified via the same
    `dependency_identity` attribute convention `memory.features.extract_feature_vector`
    already reads -- not a separate schema (Section 8.2.3's "vulnerable
    dependency identity" is the same notion of dependency this section's
    Dep(m) draws on).
    """
    neighborhood = graph.k_hop_neighborhood(mission_node, k_hop)
    return frozenset(
        n
        for n in neighborhood
        if graph.get_node(n).attributes.get("dependency_identity") is not None
    )


def dependency_subgraph_adjacency(
    graph: TemporalReasoningMemoryGraph, dep_nodes: frozenset[str]
) -> dict[str, list[str]]:
    """Directed adjacency of the dependency subgraph: edges of the TRMG
    whose source *and* target are both in `dep_nodes` ("restricted to
    dependency edges," Section 12.6.2)."""
    adjacency: dict[str, list[str]] = {n: [] for n in dep_nodes}
    for source in dep_nodes:
        for target in graph.out_edges(source):
            if target in dep_nodes:
                adjacency[source].append(target)
    return adjacency


def betweenness_centrality(adjacency: dict[str, list[str]]) -> dict[str, float]:
    """Brandes' algorithm (2001) for betweenness centrality on an
    unweighted directed graph: for every node w, the sum over all
    source/target pairs (s,t) of the fraction of shortest s->t paths that
    pass through w. O(|V|*|E|), exact (not approximated).

    Returns 0.0 for every node in a graph with no dependency-to-dependency
    edges (nothing lies "between" anything), which is the honest value,
    not an error.
    """
    centrality: dict[str, float] = dict.fromkeys(adjacency, 0.0)

    for s in adjacency:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {v: [] for v in adjacency}
        sigma: dict[str, float] = dict.fromkeys(adjacency, 0.0)
        sigma[s] = 1.0
        distance: dict[str, int] = dict.fromkeys(adjacency, -1)
        distance[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adjacency[v]:
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta: dict[str, float] = dict.fromkeys(adjacency, 0.0)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    return centrality
