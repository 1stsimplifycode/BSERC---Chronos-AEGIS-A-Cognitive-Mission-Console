"""Consensus and Conflict (manuscript Section 9.3).

    Conflict(H) = ( sum_{(H_i,H_j) in Attacks} w_i*w_j*conf_i*conf_j )
                  / ( sum_{i<j} w_i*w_j )

"which is high when credible, confident agents attack one another and low
when disagreement is confined to low-credibility or low-confidence agents."

    delta(m,t) = argmax_{claim} sum_{i : claim_i = claim, H_i in Extension(AG)} w_i*conf_i

"a credibility- and confidence-weighted aggregation restricted to the
arguments that survive mutual attack under a chosen argumentation
semantics."
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.argumentation.graph import ArgumentationGraph
from agents.hypothesis import AgentHypothesis


def conflict_score(graph: ArgumentationGraph, credibility: dict[str, float]) -> float:
    """Conflict(H) (Section 9.3). Returns 0.0 for fewer than two agents
    (there is no possible pairwise conflict to normalize by), which is the
    honest degenerate case rather than a division error."""
    agent_ids = sorted(graph.agent_ids)
    if len(agent_ids) < 2:
        return 0.0

    conf = {h.agent_id: h.confidence for h in graph.hypotheses}

    numerator = sum(credibility[a] * credibility[b] * conf[a] * conf[b] for a, b in graph.attacks)
    denominator = sum(credibility[a] * credibility[b] for a, b in combinations_pairs(agent_ids))
    return numerator / denominator if denominator > 0 else 0.0


def combinations_pairs(items: list[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


@dataclass(frozen=True)
class ConsensusResult:
    claim: str | None
    supporting_hypotheses: tuple[AgentHypothesis, ...]


def decide(
    graph: ArgumentationGraph, extension: frozenset[str], credibility: dict[str, float]
) -> ConsensusResult:
    """delta(m,t) = argmax_{claim} sum w_i*conf_i restricted to H_i in
    Extension(AG) (Section 9.3, Algorithm 4 line 5).

    Ties (equal weighted-confidence sums for distinct claims) are broken by
    taking the lexicographically greatest claim string, for full
    determinism -- the manuscript does not specify a tie-break, and this is
    a deliberate, documented choice rather than left to dict/set iteration
    order.
    """
    if not extension:
        return ConsensusResult(claim=None, supporting_hypotheses=())

    scores: dict[str, float] = {}
    supporters: dict[str, list[AgentHypothesis]] = {}
    for h in graph.hypotheses:
        if h.agent_id not in extension:
            continue
        scores[h.claim] = scores.get(h.claim, 0.0) + credibility[h.agent_id] * h.confidence
        supporters.setdefault(h.claim, []).append(h)

    best_claim = max(scores, key=lambda c: (scores[c], c))
    supporting = tuple(sorted(supporters[best_claim], key=lambda h: h.agent_id))
    return ConsensusResult(claim=best_claim, supporting_hypotheses=supporting)
