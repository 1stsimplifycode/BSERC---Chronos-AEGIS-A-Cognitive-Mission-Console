"""Historical snapshots for the TRMG (Phase 8's TRMG Evolution
requirement: "support ... historical snapshots. Every update must remain
reversible for replay.").

A `TRMGSnapshot` is a deep, point-in-time copy of a
`TemporalReasoningMemoryGraph`'s full internal state: every node's
attributes and `NodeHistory` logs, and every edge's attributes.
`TemporalReasoningMemoryGraph.snapshot()`/`.restore()` (graph/trmg/model.py)
are the only things that construct or consume one.

This is what "every update remains reversible for replay" means
concretely here: rather than making every individual mutation
(`add_node`, `add_edge`, `apply_decay`, temporal pruning, ...)
independently invertible -- a mechanism the manuscript does not specify --
a caller takes a snapshot before a batch of updates (or before a
destructive operation like pruning) and can restore to exactly that
point afterward. This is the same "replay from a known point" contract
`digital_twin/replay.py` already establishes for whole scenario runs
(Section 15), applied here at the level of a single TRMG's internal
state rather than a whole Digital Twin run.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.trmg.model import EdgeAttributes, TRMGNode


@dataclass(frozen=True)
class TRMGSnapshot:
    """An immutable, deep, point-in-time copy of a TRMG's internal state.
    `label` is a caller-supplied identifier (e.g. simulation time or a
    free-text tag) -- this module does not read any clock of its own."""

    label: str
    nodes: dict[str, TRMGNode]
    out_edges: dict[str, dict[str, EdgeAttributes]]
    in_edges: dict[str, dict[str, EdgeAttributes]]
