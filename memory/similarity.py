"""Similarity, recency weighting, and outcome confidence for Episodic
Memory Retrieval (manuscript Section 8.4, Algorithm 3).

Section 8.4: "The similarity function sim(q, s_i) is a design parameter; we
specify it as a weighted combination of exact-match indicators (identical
vulnerable-dependency identity, identical communication-topology class) and
continuous-feature distance (weather, mission-phase proximity), rather than
fully specifying its weights here, since those weights are precisely what
RQ6 and RQ7 (Section 3) are designed to calibrate empirically."

`SimilarityWeights` therefore defaults to unweighted (1.0) contributions,
matching the manuscript's explicit statement that the weights themselves
are an open calibration question (Section 8.6), not a fitted constant.

Algorithm 3, line 6 ranks candidates by
    sim(q, s_i) * recency_weight(t_i) * outcome_confidence(o_i)
`recency_weight` and the notion of an outcome carrying a confidence are
named by the manuscript but not given closed forms; this module documents
the specific, minimal forms chosen here (exponential recency decay,
caller-supplied outcome confidence) as design choices, analogous to
reasoning/trust/bayesian.py's documented modeling choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from memory.features import FeatureVector


@dataclass(frozen=True)
class SimilarityWeights:
    """Weights for sim(q, s_i) (Section 8.4). All default to 1.0: per
    Section 8.6, these require empirical calibration (RQ6, RQ7) and are not
    fitted constants here."""

    dependency_weight: float = 1.0
    topology_weight: float = 1.0
    weather_weight: float = 1.0
    phase_weight: float = 1.0
    entry_node_weight: float = 1.0
    criticality_tier_weight: float = 1.0
    continuous_weight: float = 1.0


def similarity(
    query: FeatureVector, candidate: FeatureVector, weights: SimilarityWeights | None = None
) -> float:
    """sim(q, s_i): weighted exact-match indicators + continuous-feature
    proximity (Section 8.4)."""
    weights = weights or SimilarityWeights()

    score = 0.0
    if (
        query.dependency_identity is not None
        and query.dependency_identity == candidate.dependency_identity
    ):
        score += weights.dependency_weight
    if query.topology_class is not None and query.topology_class == candidate.topology_class:
        score += weights.topology_weight
    if query.weather_category is not None and query.weather_category == candidate.weather_category:
        score += weights.weather_weight
    if query.mission_phase is not None and query.mission_phase == candidate.mission_phase:
        score += weights.phase_weight
    if query.entry_node_type is not None and query.entry_node_type == candidate.entry_node_type:
        score += weights.entry_node_weight
    if (
        query.target_criticality_tier is not None
        and query.target_criticality_tier == candidate.target_criticality_tier
    ):
        score += weights.criticality_tier_weight

    common_keys = set(query.continuous_features) & set(candidate.continuous_features)
    if common_keys:
        squared_diffs = [
            (query.continuous_features[k] - candidate.continuous_features[k]) ** 2
            for k in common_keys
        ]
        distance = math.sqrt(sum(squared_diffs) / len(squared_diffs))
        score += weights.continuous_weight / (1.0 + distance)

    return score


def recency_weight(t_i: float, now: float, decay_rate: float = 0.01) -> float:
    """recency-weight(t_i) (Algorithm 3, line 6): exponential decay with
    elapsed time since the episode, mirroring the TRMG's own confidence
    decay form (Section 6.1) for consistency rather than introducing an
    unrelated decay family.
    """
    elapsed = max(0.0, now - t_i)
    return math.exp(-decay_rate * elapsed)
