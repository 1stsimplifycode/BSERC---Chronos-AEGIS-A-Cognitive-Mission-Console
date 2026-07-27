"""AI Contribution 6: Cognitive Cyber Reasoning (manuscript Section 11).

"Cognitive Cyber Reasoning is a complementary, memory-driven mechanism:
rather than starting from a vulnerability, it starts from the current
mission's feature vector s (Section 8.2.3) and asks what happened the last
time a structurally similar situation occurred, using Episodic Memory
(Section 8.4) rather than a fresh propagation computation."

"Cognitive Cyber Reasoning's output mu(q) is combined with, not
substituted for, the propagation-based cyber impact head C(m,t) of
Section 7: we treat mu(q) as an additional input to the argumentation
layer (Section 9.1's Cyber Analyst agent), to be weighed against
propagation-based evidence rather than trusted exclusively, since
memory-driven prediction is only as good as the relevance of the
retrieved episodes (Section 8.6)."

This module deliberately does not create an independent subsystem: it
calls `cyber.consequence.predict_mission_consequence` (the same mu(q)
formula Section 12.2 reuses) against the caller's `EpisodicMissionMemory`
episodic store, and exposes a combination function that blends mu(q) with
an already-computed C(m,t) rather than replacing it -- matching "combined
with, not substituted for" literally.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyber.consequence import ConsequencePrediction, predict_mission_consequence
from memory.episodic import EpisodicEntry, EpisodicMemoryStore
from memory.features import FeatureVector
from memory.outcome import MissionOutcome
from memory.similarity import SimilarityWeights


def is_cyber_relevant_episode(entry: EpisodicEntry) -> bool:
    """The retrieval restriction Section 11 specifies: "restricted to
    episodes whose recorded decision delta or outcome o involved a cyber
    event." `ReconciledDecision` (Section 9.7) has no structured field
    marking a decision as "involving a cyber event" -- adding one is
    Section 9's concern, not Section 11's -- so this restricts on the
    outcome-based half of that condition, which is directly available as
    `MissionOutcome.CYBER_INCIDENT`.
    """
    return entry.outcome == MissionOutcome.CYBER_INCIDENT


@dataclass(frozen=True)
class CognitiveCyberReasoningResult:
    consequence: ConsequencePrediction
    combined_cyber_impact: float
    """mu(q) blended with the propagation-based C(m,t) head (Section 7),
    per Section 11's "combined with, not substituted for.\""""
    summary: str


class CognitiveCyberReasoning:
    """Section 11's memory-driven cyber-consequence predictor, wired
    directly to an `EpisodicMemoryStore` rather than a separate store."""

    def __init__(
        self,
        store: EpisodicMemoryStore,
        *,
        top_k: int = 5,
        weights: SimilarityWeights | None = None,
        recency_decay_rate: float = 0.01,
        severity: dict[MissionOutcome, float] | None = None,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.weights = weights
        self.recency_decay_rate = recency_decay_rate
        self.severity = severity

    def predict(self, query: FeatureVector, now: float) -> ConsequencePrediction:
        """mu(q) (Section 11)."""
        return predict_mission_consequence(
            query,
            self.store,
            now,
            predicate=is_cyber_relevant_episode,
            top_k=self.top_k,
            weights=self.weights,
            recency_decay_rate=self.recency_decay_rate,
            severity=self.severity,
        )

    def combine_with_propagation_head(
        self, query: FeatureVector, now: float, cyber_impact_head: float, mu_weight: float = 0.5
    ) -> CognitiveCyberReasoningResult:
        """Blend mu(q) with C(m,t) (Section 7's propagation-based cyber
        impact head).

        ASSUMPTION (documented): the manuscript says mu(q) is "weighed
        against propagation-based evidence rather than trusted
        exclusively" without giving a specific weighting; `mu_weight=0.5`
        (equal weighting) is this repository's documented default, not a
        manuscript-specified constant -- override it when calibrated
        weights become available (RQ9, Section 3).
        """
        if not 0.0 <= mu_weight <= 1.0:
            raise ValueError(f"mu_weight must be in [0,1], got {mu_weight}")

        consequence = self.predict(query, now)
        if consequence.mu is None:
            return CognitiveCyberReasoningResult(
                consequence=consequence,
                combined_cyber_impact=cyber_impact_head,
                summary=(
                    f"no memory-driven evidence available; using propagation-based "
                    f"C(m,t)={cyber_impact_head:.4f} alone"
                ),
            )

        combined = mu_weight * consequence.mu + (1.0 - mu_weight) * cyber_impact_head
        return CognitiveCyberReasoningResult(
            consequence=consequence,
            combined_cyber_impact=combined,
            summary=(
                f"combined mu(q)={consequence.mu:.4f} (weight={mu_weight}) with "
                f"propagation-based C(m,t)={cyber_impact_head:.4f} -> {combined:.4f}"
            ),
        )
