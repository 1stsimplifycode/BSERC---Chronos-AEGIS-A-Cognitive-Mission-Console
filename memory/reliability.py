"""Episodic Learning (Phase 8): non-destructive, evidence-driven
retrieval-confidence adjustment for `memory.episodic.EpisodicMemoryStore`.

Phase 8's brief: "successful missions strengthen retrieval confidence,
unsuccessful missions reduce retrieval confidence, contradictory evidence
is retained rather than overwritten, temporal ordering is preserved,
provenance remains available. No destructive updates." The manuscript
itself gives the mechanism this maps onto: Section 8.4's rank_score
already multiplies similarity by `entry.outcome_confidence`
(`memory/episodic.py`), and Algorithm 2's EMM-Update (Section 8.3) is the
natural point at which a just-concluded mission's real outcome becomes
available to compare against what was retrieved to help reason about it.

Rather than mutating a stored `EpisodicEntry.outcome_confidence` in place
(which Proposition 4's append-only discipline, Section 5.4, and this
module's own docstring both forbid), every adjustment is recorded as an
immutable, timestamped, append-only `ConfidenceAdjustment` -- the same
append-only-log pattern already used for `graph.trmg.model.NodeHistory`.
`ReliabilityLedger.effective_confidence` folds the adjustment history on
top of the entry's original, never-modified `outcome_confidence` at read
time, so the full adjustment history (who adjusted what, when, and why)
remains queryable -- this is the "provenance remains available"
requirement, not a side effect.

ASSUMPTION (documented): the manuscript does not specify a magnitude for
how much a single confirming or contradicting mission should move
retrieval confidence. This module uses a fixed, small, bounded step
(`REINFORCEMENT_STEP = 0.05`) per event, clipped so effective confidence
never leaves [0, 1] -- a conservative, explicit design choice, not a
manuscript-specified constant.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memory.outcome import MissionOutcome

if TYPE_CHECKING:
    from memory.episodic import EpisodicEntry

REINFORCEMENT_STEP = 0.05
"""Magnitude of a single confirming/contradicting adjustment (documented
assumption -- see module docstring)."""


@dataclass(frozen=True)
class ConfidenceAdjustment:
    """One immutable, append-only adjustment to an episode's *effective*
    retrieval confidence -- never a mutation of the episode itself."""

    time: float
    delta: float
    reason: str
    source_mission_id: str


class ReliabilityLedger:
    """An append-only record of confidence adjustments, keyed by episode
    object identity (`id(entry)`) so pruning (`EpisodicMemoryStore.prune_oldest`,
    which repositions surviving entries) never misaligns an adjustment
    history with the wrong episode -- entry objects themselves survive
    pruning unchanged, only their index does not.
    """

    def __init__(self) -> None:
        self._adjustments: dict[int, list[ConfidenceAdjustment]] = defaultdict(list)

    def record(self, entry: EpisodicEntry, adjustment: ConfidenceAdjustment) -> None:
        self._adjustments[id(entry)].append(adjustment)

    def history(self, entry: EpisodicEntry) -> tuple[ConfidenceAdjustment, ...]:
        """The full, ordered provenance of adjustments to `entry` --
        available for inspection, never hidden or collapsed."""
        return tuple(self._adjustments.get(id(entry), ()))

    def effective_confidence(self, entry: EpisodicEntry) -> float:
        """`entry.outcome_confidence`, as originally recorded, folded with
        every adjustment recorded since -- clipped to [0, 1]. Never
        mutates `entry`."""
        total = sum(a.delta for a in self._adjustments.get(id(entry), ()))
        return min(1.0, max(0.0, entry.outcome_confidence + total))

    def reinforce_from_outcome(
        self,
        retrieved: EpisodicEntry,
        realized_outcome: MissionOutcome,
        *,
        time: float,
        source_mission_id: str,
    ) -> ConfidenceAdjustment:
        """Section 16.1's "mission outcome -> memory update" step, applied
        to retrieval confidence specifically: `retrieved` was cited as
        evidence for a mission that has now concluded with
        `realized_outcome`. If `retrieved`'s own recorded outcome agrees,
        this was a genuinely predictive precedent -- reinforce it. If it
        disagrees, this was a misleading precedent for this query --
        weaken it. Either way `retrieved` itself is untouched; the
        disagreement is retained as evidence (contradictory episodes are
        never discarded), not overwritten.
        """
        agrees = retrieved.outcome == realized_outcome
        delta = REINFORCEMENT_STEP if agrees else -REINFORCEMENT_STEP
        reason = (
            f"retrieved episode's outcome {retrieved.outcome.value!r} "
            f"{'agreed' if agrees else 'disagreed'} with realized outcome "
            f"{realized_outcome.value!r} for mission {source_mission_id!r}"
        )
        adjustment = ConfidenceAdjustment(
            time=time, delta=delta, reason=reason, source_mission_id=source_mission_id
        )
        self.record(retrieved, adjustment)
        return adjustment
