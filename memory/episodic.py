"""Episodic Memory M_e (manuscript Section 8.2.3) and the indexed search
Algorithm 3 (Section 8.4) requires.

An episodic entry is
    e = (m_id, t, s, delta, T, A_hyp, C*, o)
M_e = {e_1, e_2, ...} accumulates one such tuple per completed mission or
mission segment (append-only, Proposition 4's discipline, Section 5.4 --
mirrored here exactly as it is on TRMG node history, graph/trmg/model.py).

IMPLEMENTATION STATUS -- indexing: Section 8.5 specifies O(log|M_e|) memory
retrieval "under an indexed approximate nearest-neighbor search (for
example, a graph-based or tree-based index over the feature space)" as the
reference design. `EpisodicMemoryStore` implements a simpler, real indexed
structure: a hash bucket keyed on the two exact-match categorical fields
(dependency_identity, topology_class), so a query with either field set
narrows the candidate set to O(1) buckets before ranking, rather than
scanning all of M_e. This is a genuine index (not a linear scan dressed up
as one), but it is not a full ANN structure (a k-d tree, ball tree, or
HNSW graph): a query whose categorical fields are both None, or whose
combination has never been seen, falls back to a full scan of M_e, and
worst-case complexity for a single bucket is O(|bucket|) rather than a
strict O(log|M_e|). This is a documented simplification, not the "not yet
implemented" placeholder pattern used elsewhere in this repository --
Algorithm 3 itself is fully implemented; only its indexing structure is
simpler than the manuscript's reference design.

PHASE 8 EXTENSIONS: `EpisodicEntry` gains an `embedding` field (Section
16.3's experience replay) and `EpisodicMemoryStore` gains a
`ReliabilityLedger` (Section 16.1's "mission outcome -> memory update"
step, applied to retrieval confidence -- see `memory/reliability.py`).
Neither changes Section 8.2.3's core tuple e = (m_id, t, s, delta, T,
A_hyp, C*, o) or Algorithm 3's ranking formula in spirit: `embedding` is
additive metadata, and `rank_score` still uses "sim * recency_weight *
outcome_confidence" exactly as before, just reading `outcome_confidence`
through the (non-destructive) reliability ledger rather than the entry's
raw, immutable field directly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agents.hypothesis import AgentHypothesis
from memory.features import FeatureVector
from memory.outcome import MissionOutcome
from memory.reliability import ReliabilityLedger
from memory.similarity import SimilarityWeights, recency_weight, similarity


@dataclass(frozen=True)
class EpisodicEntry:
    """e = (m_id, t, s, delta, T, A_hyp, C*, o) (Section 8.2.3)."""

    mission_id: str
    time: float
    features: FeatureVector
    reconciled_decision: Any
    trust: float
    agent_hypotheses: tuple[AgentHypothesis, ...]
    causal_context: Any
    outcome: MissionOutcome
    outcome_confidence: float = 1.0
    embedding: tuple[float, ...] = ()
    """h_m(t) (Section 7.1), the mission-aware GNN's attention embedding at
    the time this episode was recorded -- not part of Section 8.2.3's
    tuple e = (m_id, t, s, delta, T, A_hyp, C*, o), but required for
    genuine experience replay (Section 16.3): `s` (`features`) is the
    categorical/continuous summary Algorithm 3 retrieves *by*, whereas
    `embedding` is the actual numeric input the prediction heads (theta)
    consume, which Episodic Memory would otherwise have no way to
    reconstruct for a past mission. Empty by default (backward compatible
    with entries recorded before this field existed, or recorded by a
    caller with no embedding to offer) -- an empty embedding simply means
    that entry is not eligible for experience replay
    (`reasoning.continual.algorithm7`), not an error.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.outcome_confidence <= 1.0:
            raise ValueError(f"outcome_confidence must be in [0,1], got {self.outcome_confidence}")


@dataclass(frozen=True)
class RankedEpisodicCandidate:
    """One row of Algorithm 3's ranked candidate list (line 6)."""

    entry: EpisodicEntry
    similarity: float
    rank_score: float  # sim * recency_weight * outcome_confidence


def _bucket_key(features: FeatureVector) -> tuple[str | None, str | None]:
    return (features.dependency_identity, features.topology_class)


class EpisodicMemoryStore:
    """M_e: the append-only episodic entry store, plus Algorithm 3's
    IndexedSimilaritySearch."""

    def __init__(self) -> None:
        self._entries: list[EpisodicEntry] = []
        self._index: dict[tuple[str | None, str | None], list[int]] = defaultdict(list)
        self.reliability = ReliabilityLedger()
        """Non-destructive, outcome-driven retrieval-confidence adjustments
        (Phase 8's Episodic Learning requirement) -- see
        `memory/reliability.py`. Adjustments are keyed by episode object
        identity, so they survive `prune_oldest`'s reindexing."""

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, entry: EpisodicEntry) -> None:
        """Algorithm 2, line 5: append e to M_e. O(1) amortized (Section 8.5)."""
        index = len(self._entries)
        self._entries.append(entry)
        self._index[_bucket_key(entry.features)].append(index)

    def all_entries(self) -> tuple[EpisodicEntry, ...]:
        return tuple(self._entries)

    def prune_oldest(self, keep: int) -> int:
        """Retention-bound pruning (Algorithm 2, line 11): keep only the
        `keep` most recent entries by `time`, discarding the rest. Returns
        the number of entries removed.

        This particular pruning policy (oldest-by-time first) is a design
        choice this module makes explicitly, not one the manuscript
        specifies: Section 6.4 and Section 8.6 both flag "how aggressively
        to prune or summarize old entries" as an open empirical question.
        """
        if keep >= len(self._entries):
            return 0
        surviving = sorted(self._entries, key=lambda e: e.time, reverse=True)[:keep]
        removed = len(self._entries) - len(surviving)
        self._entries = sorted(surviving, key=lambda e: e.time)
        self._index = defaultdict(list)
        for index, entry in enumerate(self._entries):
            self._index[_bucket_key(entry.features)].append(index)
        return removed

    def search(
        self,
        query: FeatureVector,
        top_k: int,
        now: float,
        weights: SimilarityWeights | None = None,
        recency_decay_rate: float = 0.01,
        predicate: Callable[[EpisodicEntry], bool] | None = None,
    ) -> tuple[RankedEpisodicCandidate, ...]:
        """Algorithm 3 (Section 8.4): IndexedSimilaritySearch + ranking.

        Falls back to a full scan (documented in the module docstring) when
        the query's exact-match bucket is empty or unset, rather than
        returning nothing for a query the index cannot narrow.

        `predicate`, when supplied, restricts the candidate set to entries
        for which it returns True *before* ranking -- this is how Section
        11 ("restricted to episodes whose recorded decision or outcome
        involved a cyber event") and Section 12.2 ("restricted to
        attack-chain-shaped episodes") reuse this same Algorithm 3
        implementation rather than each defining their own retrieval, per
        `cyber/consequence.py`.
        """
        bucket = self._index.get(_bucket_key(query), [])
        candidate_indices = bucket if bucket else range(len(self._entries))
        if predicate is not None:
            candidate_indices = [i for i in candidate_indices if predicate(self._entries[i])]

        ranked: list[RankedEpisodicCandidate] = []
        for i in candidate_indices:
            entry = self._entries[i]
            sim = similarity(query, entry.features, weights)
            effective_confidence = self.reliability.effective_confidence(entry)
            rank_score = (
                sim * recency_weight(entry.time, now, recency_decay_rate) * effective_confidence
            )
            ranked.append(
                RankedEpisodicCandidate(entry=entry, similarity=sim, rank_score=rank_score)
            )

        ranked.sort(key=lambda c: c.rank_score, reverse=True)
        return tuple(ranked[:top_k])
