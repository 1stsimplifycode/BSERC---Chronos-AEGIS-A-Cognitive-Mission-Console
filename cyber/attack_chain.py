"""Cybersecurity Contribution 1, Section 12.2: AI-Driven Attack Chain
Reasoning.

"We extend base propagation with reasoning over Episodic Memory: rather
than treating each attack chain as a fresh graph traversal, the framework
retrieves (Algorithm 3) prior missions in which a structurally similar
attack chain -- same entry node type, same intermediate dependency, same
target criticality tier -- was observed, and uses the recorded outcome of
that chain to adjust the propagated risk estimate, in the same manner as
Section 11's Cognitive Cyber Reasoning but specifically restricted to
attack-chain-shaped episodes rather than general cyber episodes."

"In the same manner as Section 11" is implemented literally: this module
calls the same `cyber.consequence.predict_mission_consequence` mu(q)
formula Section 11 uses, and blends its result with the base propagation
estimate (`cyber.risk_propagation`, Section 12.1) the same way Section 11
blends mu(q) with C(m,t) -- one blending mechanism, reused, not two.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyber.consequence import ConsequencePrediction, predict_mission_consequence
from memory.episodic import EpisodicEntry, EpisodicMemoryStore
from memory.features import FeatureVector
from memory.outcome import MissionOutcome
from memory.similarity import SimilarityWeights


def is_attack_chain_episode(entry: EpisodicEntry) -> bool:
    """Section 12.2's retrieval restriction: "attack-chain-shaped
    episodes" are identified by `entry_node_type` being populated, since
    only an attack-path-derived feature vector sets it (see
    `cyber.attack_graph.attack_path_feature_vector`) -- an ordinary
    mission episode's feature vector (Section 8.2.3) never does.
    """
    return entry.features.entry_node_type is not None


@dataclass(frozen=True)
class AttackChainRiskAdjustment:
    """The result of adjusting a propagation-based risk estimate with
    memory-driven evidence about structurally similar past attack chains."""

    base_attenuation: float
    consequence: ConsequencePrediction
    adjusted_risk: float
    summary: str


class AttackChainReasoning:
    """Section 12.2's memory-driven attack-chain risk adjuster."""

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
        """mu(q), restricted to attack-chain-shaped episodes (Section 12.2)."""
        return predict_mission_consequence(
            query,
            self.store,
            now,
            predicate=is_attack_chain_episode,
            top_k=self.top_k,
            weights=self.weights,
            recency_decay_rate=self.recency_decay_rate,
            severity=self.severity,
        )

    def adjust_propagated_risk(
        self,
        query: FeatureVector,
        now: float,
        base_attenuation: float,
        mu_weight: float = 0.5,
    ) -> AttackChainRiskAdjustment:
        """Blend the propagation-based attenuation estimate (Section 12.1,
        `cyber.risk_propagation.attenuation_to`) with memory-driven mu(q)
        for structurally similar past attack chains -- "in the same
        manner as Section 11" (see `cyber.cognitive_reasoning`'s identical
        blend for the same documented `mu_weight=0.5` default rationale).
        """
        if not 0.0 <= mu_weight <= 1.0:
            raise ValueError(f"mu_weight must be in [0,1], got {mu_weight}")

        consequence = self.predict(query, now)
        if consequence.mu is None:
            return AttackChainRiskAdjustment(
                base_attenuation=base_attenuation,
                consequence=consequence,
                adjusted_risk=base_attenuation,
                summary=(
                    "no attack-chain-shaped episodes retrieved; using base "
                    f"propagation estimate {base_attenuation:.4f} unchanged"
                ),
            )

        adjusted = mu_weight * consequence.mu + (1.0 - mu_weight) * base_attenuation
        return AttackChainRiskAdjustment(
            base_attenuation=base_attenuation,
            consequence=consequence,
            adjusted_risk=adjusted,
            summary=(
                f"adjusted base attenuation {base_attenuation:.4f} with attack-chain "
                f"mu(q)={consequence.mu:.4f} (weight={mu_weight}) -> {adjusted:.4f}"
            ),
        )
