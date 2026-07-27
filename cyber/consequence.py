"""The similarity-weighted mission-consequence aggregate shared by Section
11 (Cognitive Cyber Reasoning) and Section 12.2 (AI-Driven Attack Chain
Reasoning).

Section 11: "Formally, given the current mission's feature vector q
(vulnerable-dependency identity, communication topology, weather, mission
phase), Cognitive Cyber Reasoning retrieves the top-k most similar past
episodes via Algorithm 3 restricted to episodes whose recorded decision
delta or outcome o involved a cyber event, and predicts the current
mission's cyber consequence as a similarity-weighted aggregate of those
episodes' recorded outcomes,

    mu(q) = sum_{i=1}^{k} [ sim(q,s_i) * outcome-confidence(o_i) * o_i ]
            / sum_{i=1}^{k} [ sim(q,s_i) * outcome-confidence(o_i) ]

where mu(q) is the predicted mission consequence."

Section 12.2: "We extend base propagation with reasoning over Episodic
Memory: rather than treating each attack chain as a fresh graph traversal,
the framework retrieves (Algorithm 3) prior missions in which a
structurally similar attack chain... was observed, and uses the recorded
outcome of that chain to adjust the propagated risk estimate, in the same
manner as Section 11's Cognitive Cyber Reasoning but specifically
restricted to attack-chain-shaped episodes rather than general cyber
episodes."

"In the same manner as Section 11" is implemented literally here: both
sections call `predict_mission_consequence`, differing only in the
`predicate` they pass to restrict which episodes are eligible (see
cyber/cognitive_reasoning.py and cyber/attack_chain.py) -- there is one
formula, not two independent reimplementations of it.

ASSUMPTION (documented): mu(q)'s formula averages o_i directly, which
requires o_i to be numeric; Section 8.2.3 defines the outcome o as one of
five *categorical* values (success, failure, delay, cyber incident,
other), not a scale. The manuscript does not specify the numeric encoding
this formula's o_i presupposes. `DEFAULT_CONSEQUENCE_SEVERITY` below is
this repository's derived, documented, and overridable choice: a
monotonic severity scale from 0.0 (success: no adverse cyber consequence)
to 1.0 (cyber incident: the maximal cyber consequence), consistent with
mu(q) being described as "the predicted mission consequence" of a cyber
event. RQ9 (Section 3) is the empirical test of Cognitive Cyber Reasoning
generally; this severity scale is exactly the kind of calibratable
parameter such an evaluation would need to validate, not a claim of
correctness made here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from memory.episodic import EpisodicMemoryStore, RankedEpisodicCandidate
from memory.features import FeatureVector
from memory.outcome import MissionOutcome
from memory.similarity import SimilarityWeights

DEFAULT_CONSEQUENCE_SEVERITY: dict[MissionOutcome, float] = {
    MissionOutcome.SUCCESS: 0.0,
    MissionOutcome.DELAY: 0.25,
    MissionOutcome.OTHER: 0.5,
    MissionOutcome.FAILURE: 0.75,
    MissionOutcome.CYBER_INCIDENT: 1.0,
}


@dataclass(frozen=True)
class ConsequencePrediction:
    """mu(q) together with the evidence it was computed from -- so a
    caller (and the Explainability Pipeline, Section 10.7) can always
    trace the prediction back to the specific retrieved episodes."""

    mu: float | None
    """None when no episodes matched the predicate/query -- an honest
    "no evidence available" result, not a fabricated default."""
    supporting_episodes: tuple[RankedEpisodicCandidate, ...] = field(default_factory=tuple)
    summary: str = ""


def predict_mission_consequence(
    query: FeatureVector,
    store: EpisodicMemoryStore,
    now: float,
    *,
    predicate: Callable[..., bool],
    top_k: int = 5,
    weights: SimilarityWeights | None = None,
    recency_decay_rate: float = 0.01,
    severity: dict[MissionOutcome, float] | None = None,
) -> ConsequencePrediction:
    """mu(q): retrieve via Algorithm 3 restricted by `predicate`, then
    compute the similarity- and confidence-weighted average severity of
    the retrieved episodes' outcomes.
    """
    severity = severity or DEFAULT_CONSEQUENCE_SEVERITY
    candidates = store.search(
        query,
        top_k=top_k,
        now=now,
        weights=weights,
        recency_decay_rate=recency_decay_rate,
        predicate=predicate,
    )

    if not candidates:
        return ConsequencePrediction(
            mu=None, summary="no matching episodes retrieved; mu(q) is undefined"
        )

    numerator = 0.0
    denominator = 0.0
    for candidate in candidates:
        entry = candidate.entry
        weight = candidate.similarity * entry.outcome_confidence
        numerator += weight * severity[entry.outcome]
        denominator += weight

    if denominator == 0.0:
        return ConsequencePrediction(
            mu=None,
            supporting_episodes=candidates,
            summary=(
                f"{len(candidates)} episode(s) retrieved but all had zero "
                f"similarity*confidence weight; mu(q) is undefined"
            ),
        )

    mu = numerator / denominator
    return ConsequencePrediction(
        mu=mu,
        supporting_episodes=candidates,
        summary=f"mu(q)={mu:.4f} from {len(candidates)} retrieved episode(s)",
    )
