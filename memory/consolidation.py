"""Batched Consolidation (manuscript Section 8.3.1).

"To avoid the O(|M_e|) worst-case cost noted in Section 5.3, consolidation
(lines 6-9 and line 11 [of Algorithm 2]) is run on a fixed schedule (for
example, after every N completed missions) rather than on every query,
amortizing its cost across N queries; between consolidation runs, new
episodes are appended to M_e in O(1) and the semantic-rule distillation and
retention-bound pruning are deferred."

`ConsolidationSchedule` tracks the "every N completed missions" trigger;
`consolidate` performs the deferred work itself (semantic distillation,
meta-memory recompute, retention-bound pruning) in one batched pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory.episodic import EpisodicMemoryStore
from memory.meta import MetaMemory
from memory.semantic import SemanticMemory, SemanticRule


@dataclass(frozen=True)
class ConsolidationConfig:
    """Batching parameters (Section 8.3, Section 8.3.1)."""

    interval_missions: int = 10  # N: run consolidation every N completed missions
    retention_bound: int = 1000  # Algorithm 2, line 11
    theta_pattern: int = 3  # theta_pattern: Algorithm 2, line 6

    def __post_init__(self) -> None:
        if self.interval_missions < 1:
            raise ValueError("interval_missions must be >= 1")
        if self.retention_bound < 1:
            raise ValueError("retention_bound must be >= 1")
        if self.theta_pattern < 1:
            raise ValueError("theta_pattern must be >= 1")


@dataclass(frozen=True)
class ConsolidationReport:
    updated_rules: tuple[SemanticRule, ...]
    pruned_count: int
    entries_considered: int


class ConsolidationSchedule:
    """Tracks completed-mission count since the last consolidation run and
    decides when Algorithm 2 lines 6-9/11 are due."""

    def __init__(self, config: ConsolidationConfig) -> None:
        self.config = config
        self._missions_since_last = 0

    def record_completed_mission(self) -> None:
        self._missions_since_last += 1

    def due(self, store_size: int) -> bool:
        """Section 8.3.1: due on the fixed N-mission schedule, or
        immediately if the retention bound is already exceeded (Algorithm
        2, line 11's condition) regardless of the schedule."""
        return (
            self._missions_since_last >= self.config.interval_missions
            or store_size > self.config.retention_bound
        )

    def reset(self) -> None:
        self._missions_since_last = 0


def consolidate(
    store: EpisodicMemoryStore,
    semantic: SemanticMemory,
    meta: MetaMemory,
    config: ConsolidationConfig,
) -> ConsolidationReport:
    """Algorithm 2, lines 6-9 and line 11, run as one batched pass."""
    entries = store.all_entries()
    updated_rules = semantic.distill(entries, config.theta_pattern)
    meta.recompute(entries)
    pruned = 0
    if len(store) > config.retention_bound:
        pruned = store.prune_oldest(keep=config.retention_bound)
    return ConsolidationReport(
        updated_rules=updated_rules, pruned_count=pruned, entries_considered=len(entries)
    )
