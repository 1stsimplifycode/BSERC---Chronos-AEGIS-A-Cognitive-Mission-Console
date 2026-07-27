#!/usr/bin/env python
"""Example: Structured AI Argumentation (Section 9) in action.

Demonstrates `StructuredArgumentAggregator` (agents/argumentation/aggregation.py,
Algorithm 4) wired into `TTRNPipeline` in place of the default
`NullArgumentationReconciler`, across two scenarios:

  1. A resolvable disagreement: Aircraft Engineer proposes "delay the
     patch until after the satellite pass" and attacks Cyber Analyst's
     "patch now" claim; Cyber Analyst does not attack back. Aircraft
     Engineer is unattacked, so it is in the grounded extension; Cyber
     Analyst is attacked by an undefeated Aircraft Engineer, so it is
     excluded. Algorithm 4 resolves this cleanly at line 4 without needing
     to fall back to the preferred semantics.
  2. A genuinely unresolvable disagreement: three agents attack each other
     in a cycle (Aircraft Engineer -> Cyber Analyst -> Safety Officer ->
     Aircraft Engineer). No argument set is jointly acceptable even under
     the preferred semantics, so Algorithm 4 escalates to a human operator
     (lines 6-8) instead of silently picking a default.

Run with:
    python -m examples.run_ttrn_pipeline_with_argumentation
"""

from __future__ import annotations

import logging

import numpy as np

from agents.argumentation.aggregation import StructuredArgumentAggregator
from agents.hypothesis import AgentHypothesis
from agents.roster import AIRCRAFT_ENGINEER, CYBER_ANALYST, SAFETY_OFFICER
from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.meta import MetaMemory
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


class FixedHypothesisGenerator:
    """A minimal HypothesisGeneratorInterface that always returns the same
    hypothesis, standing in for one of Section 9.1's nine domain agents
    (whose actual reasoning logic is not specified by the manuscript and is
    not implemented here -- see agents/roster.py)."""

    def __init__(self, agent_id: str, claim: str, confidence: float, counter: tuple[str, ...] = ()):
        self.agent_id = agent_id
        self._claim = claim
        self._confidence = confidence
        self._counter = counter

    def generate(self, neighborhood_node_ids, trust_posterior, retrieved_memory, domain_evidence):
        return AgentHypothesis(
            agent_id=self.agent_id,
            claim=self._claim,
            evidence=tuple(sorted(neighborhood_node_ids))[:2],
            confidence=self._confidence,
            counterarguments=self._counter,
        )


def build_graph() -> TemporalReasoningMemoryGraph:
    graph = TemporalReasoningMemoryGraph()
    graph.add_node("engine_health")
    graph.add_node("mission_alpha")
    graph.add_edge(
        "engine_health",
        "mission_alpha",
        timestamp=0.0,
        confidence=0.9,
        decay_rate=0.03,
        priority=0.9,
    )
    return graph


def run(generators: tuple[FixedHypothesisGenerator, ...], label: str) -> None:
    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    graph = build_graph()
    aggregator = StructuredArgumentAggregator(meta_memory=MetaMemory())
    pipeline = TTRNPipeline(
        graph, gnn, k_hop=6, hypothesis_generators=generators, argumentation=aggregator
    )

    rng = np.random.default_rng(0)
    node_features = {"engine_health": rng.normal(size=8), "mission_alpha": rng.normal(size=8)}
    result = pipeline.infer("mission_alpha", now=0.0, node_features=node_features)

    print(f"\n=== {label} ===")
    decision = result.reconciled_decision
    assert decision is not None
    if decision.escalated:
        print(f"ESCALATED to human operator: {decision.summary}")
    else:
        print(f"decision: '{decision.claim}'")
        print(f"accepted into Extension(AG): {[h.agent_id for h in decision.extension_hypotheses]}")
    print(f"Conflict(H) = {decision.conflict_score:.4f}")
    print(f"credibility weights = {decision.credibility_weights}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    run(
        (
            FixedHypothesisGenerator(
                AIRCRAFT_ENGINEER,
                "delay the patch until after the satellite pass",
                0.8,
                counter=(CYBER_ANALYST,),
            ),
            FixedHypothesisGenerator(CYBER_ANALYST, "patch now", 0.7),
        ),
        "Scenario 1: resolvable one-directional attack",
    )

    run(
        (
            FixedHypothesisGenerator(AIRCRAFT_ENGINEER, "delay", 0.8, counter=(CYBER_ANALYST,)),
            FixedHypothesisGenerator(CYBER_ANALYST, "patch now", 0.8, counter=(SAFETY_OFFICER,)),
            FixedHypothesisGenerator(
                SAFETY_OFFICER, "ground the aircraft", 0.8, counter=(AIRCRAFT_ENGINEER,)
            ),
        ),
        "Scenario 2: three-way mutual attack (unresolvable)",
    )


if __name__ == "__main__":
    main()
