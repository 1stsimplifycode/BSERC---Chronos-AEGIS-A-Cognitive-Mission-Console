# graph/ -- Temporal Trust Graph / Temporal Reasoning Memory Graph (Section 6)

**Status: implemented.** `graph/trmg/` is the shared graph substrate
every other CMI component (the GNN, argumentation, the causal engine,
cyber reasoning, the scheduler, the Digital Twin) reads from and writes
to -- there is exactly one TRMG per mission, never a parallel copy (see
`digital_twin/README.md`'s "No duplicated state" principle for the
canonical example of this discipline in practice).

| Section | Component | File |
|---|---|---|
| 6.1 | Temporal Trust Graph (TTG): `chi(e) = (t, w, c, r, lambda, d, pi)` | `graph/trmg/model.py` `EdgeAttributes` |
| 6.2 | TRMG extension: six append-only history logs `rho(n)` per node | `graph/trmg/model.py` `NodeHistory`, `TRMGNode` |
| 5.2 (Algorithm 1, line 1) | Decay + telemetry ingestion | `graph/trmg/model.py` `apply_decay`, `ingest_telemetry_edge` |
| 5.2 (Algorithm 1, line 2) | k-hop neighborhood traversal | `graph/trmg/model.py` `k_hop_neighborhood` |
| 6.5 | Temporal Dependency Analysis: `Vol(n,t,w)` | `graph/trmg/temporal.py` `TemporalDependencyAnalyzer` |
| Phase 8 (TRMG Evolution) | Historical snapshots, reversible restore, temporal pruning | `graph/trmg/snapshot.py` `TRMGSnapshot`; `graph/trmg/model.py` `.snapshot()`/`.restore()`/`.prune_history()`, `NodeHistory.prune()` |

## Usage

```python
from graph.trmg.model import TemporalReasoningMemoryGraph

graph = TemporalReasoningMemoryGraph()
graph.add_node("engine_health")
graph.add_node("mission")
graph.add_edge("engine_health", "mission", timestamp=0.0, confidence=0.9, decay_rate=0.05)

graph.apply_decay(now=10.0)                       # Algorithm 1, line 1
neighborhood = graph.k_hop_neighborhood("mission", k=6)   # Algorithm 1, line 2

node = graph.get_node("mission")
node.history.append_trust(10.0, 0.85)              # rho_trust(n), Section 6.2
```

Every other CMI package builds on this API directly rather than
maintaining its own graph traversal or history log: `reasoning/trust/gnn.py`'s
attention mechanism reads `in_edges`; `memory/features.py`'s
`extract_feature_vector` reads node `attributes` and (Phase 8)
incoming-edge confidence; `cyber/risk_propagation.py`'s widest-path
attenuation and `cyber/centrality.py`'s betweenness centrality both
traverse the same adjacency structure; `digital_twin/events.py`'s twelve
event types mutate the graph via `ingest_telemetry_edge`/`add_edge`
exactly as any other telemetry source would.

## Design notes

**Append-only history, enforced by the type, not convention.**
`NodeHistory` has `append_*` methods only -- no `update_*`/`remove_*` --
so Proposition 4 (Section 5.4)'s append-only discipline cannot be
violated by construction, mirrored exactly by
`memory/episodic.py`'s `EpisodicMemoryStore.append`.

**Snapshots are deep copies, not references (Phase 8).** `snapshot()`
captures the full internal state (`copy.deepcopy` of every node and
edge); `restore()` deep-copies again on the way back in, so a snapshot
can be restored from more than once and mutating the live graph after a
restore never corrupts the snapshot. `NodeHistory.prune()` is the one
operation in this package that is not itself reversible in place --
`docs/continual-learning-guide.md` and `graph/trmg/model.py`'s own
docstring cover the "snapshot before pruning" workflow that makes it
reversible for replay in practice.

**No vulnerability database, no ANN index.** Section 6's formalism does
not require either; where downstream packages need something like them
(`cyber/`'s intrinsic risk values, `memory/`'s indexed similarity
search), each documents its own honest simplification rather than this
package fabricating one on their behalf.

## Tests

`tests/graph/test_trmg.py` (TTG/TRMG core), `test_temporal.py` (Temporal
Dependency Analysis), `test_snapshot.py` (Phase 8 snapshot/restore/
pruning) -- 97-100% line coverage; see `docs/code-quality-audit.md` for
the handful of uncovered defensive branches and why they were judged
acceptable.
