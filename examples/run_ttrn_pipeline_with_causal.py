#!/usr/bin/env python
"""Example: Causal Mission Intelligence (Section 10) in action, and the
eight-level Explainability Pipeline (Section 10.7) assembled over a full
Algorithm 1 run that also uses real Episodic Mission Memory (Section 8)
and Structured AI Argumentation (Section 9) -- the capstone example
showing every implemented CMI component working together.

Run with:
    python -m examples.run_ttrn_pipeline_with_causal
"""

from __future__ import annotations

import logging

import numpy as np

from agents.argumentation.aggregation import StructuredArgumentAggregator
from agents.hypothesis import AgentHypothesis
from agents.roster import AIRCRAFT_ENGINEER, CYBER_ANALYST
from graph.trmg.model import TemporalReasoningMemoryGraph
from graph.trmg.temporal import TemporalDependencyAnalyzer
from memory.emm import EpisodicMissionMemory
from reasoning.causal.algorithm5 import CandidateIntervention, StructuralCausalEngine
from reasoning.causal.explainability import ExplanationInputs, assemble_explanation
from reasoning.causal.scm import StructuralCausalModel
from reasoning.trust.criticality import MissionCriticalityModel
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


class FixedHypothesisGenerator:
    def __init__(self, agent_id: str, claim: str, confidence: float):
        self.agent_id = agent_id
        self._claim = claim
        self._confidence = confidence

    def generate(self, neighborhood_node_ids, trust_posterior, retrieved_memory, domain_evidence):
        return AgentHypothesis(
            agent_id=self.agent_id, claim=self._claim, evidence=(), confidence=self._confidence
        )


def build_graph() -> TemporalReasoningMemoryGraph:
    graph = TemporalReasoningMemoryGraph()
    graph.add_node("engine_health")
    graph.add_node("weather")
    graph.add_node("mission_alpha")
    graph.add_edge(
        "engine_health",
        "mission_alpha",
        timestamp=0.0,
        confidence=0.9,
        decay_rate=0.03,
        priority=0.9,
    )
    graph.add_edge(
        "weather", "mission_alpha", timestamp=0.0, confidence=0.6, decay_rate=0.05, priority=0.5
    )
    return graph


def build_scm() -> StructuralCausalModel:
    """A domain-supplied (not discovered -- Section 10.4) SCM: comm
    availability causally affects mission trust."""
    scm = StructuralCausalModel()
    scm.register("comm_availability", parents=(), function=lambda p, u: u, exogenous=0.85)
    scm.register(
        "mission_trust",
        parents=("comm_availability",),
        function=lambda p, u: 0.5 * p["comm_availability"] + u,
        exogenous=0.1,
    )
    return scm


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    graph = build_graph()
    dependency_analyzer = TemporalDependencyAnalyzer(graph)
    dependency_analyzer.record_snapshot("mission_alpha", 0.0)

    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    rng = np.random.default_rng(0)
    node_features = {
        "engine_health": rng.normal(size=8),
        "weather": rng.normal(size=8),
        "mission_alpha": rng.normal(size=8),
    }

    memory = EpisodicMissionMemory()
    argumentation = StructuredArgumentAggregator(meta_memory=memory.meta)
    scm = build_scm()
    causal_engine = StructuralCausalEngine(scm, graph, gnn, "mission_alpha", node_features, now=0.0)

    generators = (
        FixedHypothesisGenerator(AIRCRAFT_ENGINEER, "proceed with mission", 0.85),
        FixedHypothesisGenerator(CYBER_ANALYST, "proceed with mission", 0.6),
    )
    pipeline = TTRNPipeline(
        graph,
        gnn,
        k_hop=6,
        memory=memory,
        hypothesis_generators=generators,
        argumentation=argumentation,
        causal_engine=causal_engine,
    )

    candidates = (
        CandidateIntervention(variable="comm_availability", value=0.0),  # causal branch
        CandidateIntervention(variable="weather->mission_alpha", value=0.0),  # correlational branch
    )
    result = pipeline.infer(
        "mission_alpha", now=0.0, node_features=node_features, candidate_interventions=candidates
    )

    print("=== Algorithm 5: Causal Counterfactual Generation ===")
    for r in result.causal_context.results:
        print(
            f"  do({r.intervention.variable}={r.intervention.value}): "
            f"delta_T={r.delta_trust:+.4f}  [{r.causal_status}]"
        )

    criticality = MissionCriticalityModel(alpha_base=0.5)
    kappa_m = 0.7
    trust_floor = criticality.trust_floor(kappa_m)
    volatility = dependency_analyzer.volatility("mission_alpha", 0.0, window=10.0)

    decision = result.reconciled_decision
    inputs = ExplanationInputs(
        mission_id="mission_alpha",
        trust_posterior=result.posterior.mean,
        uncertainty=result.posterior.variance,
        neighborhood_size=len(graph.k_hop_neighborhood("mission_alpha", 6)),
        hypotheses=result.hypotheses,
        credibility_weights=decision.credibility_weights if decision else {},
        extension_agent_ids=(
            frozenset(h.agent_id for h in decision.extension_hypotheses) if decision else None
        ),
        dependency_volatility=volatility,
        trust_floor=trust_floor,
        mission_criticality=kappa_m,
        retrieved_episodes=result.state.retrieved_memory,
        causal_results=result.causal_context.results,
    )
    report = assemble_explanation(inputs)

    print("\n=== Section 10.7: Eight-Level Explainability Pipeline ===")
    for level in report.levels:
        marker = "" if level.available else " (unavailable)"
        print(f"[{level.name}]{marker} {level.content}")


if __name__ == "__main__":
    main()
