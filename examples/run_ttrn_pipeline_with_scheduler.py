#!/usr/bin/env python
"""Example: the AI Patch Scheduler (Section 14, Algorithm 6), consuming
*real* CyberExposure (Algorithm 6 lines 1-2, via Section 12.6.1's MTES,
`cyber.scheduler_integration`) and *real* MissionTrustLoss (Algorithm 6
lines 3-4, via Section 10.5's causal engine) -- the full capstone: real
Episodic Mission Memory (Section 8), Structured AI Argumentation (Section
9), Causal Mission Intelligence (Section 10), Cognitive Cyber Reasoning
and Mission-Aware Cyber Risk Propagation (Sections 11-12), and AI Patch
Scheduling (Section 14) all composed in one Algorithm 1 run. No hardcoded
or placeholder cost terms remain (Phase 6).

Reproduces the manuscript's own worked illustrative scenario (Section
14.1): "a mission starting in 12 minutes, a patch requiring 4 minutes, and
a communication window closing in 6 minutes" -- the manuscript states the
*recommended* action for that scenario ("delay the patch until after the
satellite pass") but is explicit that this "remains a worked example of
the query form, not a validated policy." Both of Algorithm 6's cost terms
are computed for real here: whichever action the weighted argmin actually
picks is printed, not assumed.

Run with:
    python -m examples.run_ttrn_pipeline_with_scheduler
"""

from __future__ import annotations

import logging

import numpy as np

from agents.argumentation.aggregation import StructuredArgumentAggregator
from agents.hypothesis import AgentHypothesis
from agents.roster import AIRCRAFT_ENGINEER, CYBER_ANALYST
from cyber.cognitive_reasoning import CognitiveCyberReasoning
from cyber.metrics import mission_dependency_risk_index
from cyber.scheduler_integration import compute_cyber_exposure
from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.emm import EpisodicMissionMemory
from memory.features import FeatureVector
from reasoning.causal.algorithm5 import CandidateIntervention, StructuralCausalEngine
from reasoning.causal.explainability import ExplanationInputs, assemble_explanation
from reasoning.causal.scm import StructuralCausalModel
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline
from scheduler.algorithm6 import AIPatchScheduler
from scheduler.interfaces import DELAY, PATCH_NOW, PatchSchedulingRequest


class FixedHypothesisGenerator:
    def __init__(self, agent_id: str, claim: str, confidence: float):
        self.agent_id = agent_id
        self._claim = claim
        self._confidence = confidence

    def generate(self, neighborhood_node_ids, trust_posterior, retrieved_memory, domain_evidence):
        return AgentHypothesis(
            agent_id=self.agent_id, claim=self._claim, evidence=(), confidence=self._confidence
        )


VULNERABLE_FIRMWARE = "satcom_firmware"


def build_graph() -> TemporalReasoningMemoryGraph:
    graph = TemporalReasoningMemoryGraph()
    graph.add_node("engine_health")
    graph.add_node("satcom")
    graph.add_node(VULNERABLE_FIRMWARE, dependency_identity="satcom_fw@3.2")
    graph.add_node("mission_alpha")
    graph.add_edge(
        "engine_health",
        "mission_alpha",
        timestamp=0.0,
        confidence=0.9,
        decay_rate=0.03,
        priority=0.8,
    )
    graph.add_edge(
        "satcom", "mission_alpha", timestamp=0.0, confidence=0.7, decay_rate=0.05, priority=0.6
    )
    # The vulnerable firmware's risk propagates into satcom, then into the
    # mission -- the path Section 12.1's base propagation (reused by
    # cyber.scheduler_integration's MTES computation below) traces.
    graph.add_edge(
        VULNERABLE_FIRMWARE, "satcom", timestamp=0.0, confidence=0.85, decay_rate=0.08, priority=0.9
    )
    return graph


def build_scm() -> StructuralCausalModel:
    """A domain-supplied SCM (Section 10.4: causal discovery is not
    implemented): applying the patch now degrades communication trust
    immediately (the maintenance window overlaps the satellite pass),
    which is what makes MissionTrustLoss(patch_now) nontrivial below."""
    scm = StructuralCausalModel()
    scm.register("patch_status", parents=(), function=lambda p, u: u, exogenous=0.0)
    scm.register(
        "mission_trust",
        parents=("patch_status",),
        function=lambda p, u: 0.9 - 0.3 * p["patch_status"] + u,
        exogenous=0.0,
    )
    return scm


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    graph = build_graph()
    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    rng = np.random.default_rng(0)
    node_features = {
        "engine_health": rng.normal(size=8),
        "satcom": rng.normal(size=8),
        VULNERABLE_FIRMWARE: rng.normal(size=8),
        "mission_alpha": rng.normal(size=8),
    }

    memory = EpisodicMissionMemory()
    argumentation = StructuredArgumentAggregator(meta_memory=memory.meta)
    scm = build_scm()
    causal_engine = StructuralCausalEngine(scm, graph, gnn, "mission_alpha", node_features, now=0.0)
    scheduler = AIPatchScheduler()

    generators = (
        FixedHypothesisGenerator(AIRCRAFT_ENGINEER, "apply patch this window", 0.7),
        FixedHypothesisGenerator(CYBER_ANALYST, "apply patch this window", 0.8),
    )
    pipeline = TTRNPipeline(
        graph,
        gnn,
        k_hop=6,
        memory=memory,
        hypothesis_generators=generators,
        argumentation=argumentation,
        causal_engine=causal_engine,
        scheduler=scheduler,
    )

    # Algorithm 6, lines 3-4: MissionTrustLoss(a) via do(patch_status) at
    # the two decision points -- computed for real, not hardcoded.
    trust_loss = causal_engine.generate_counterfactual(
        None,
        (
            CandidateIntervention(variable="patch_status", value=1.0),  # patch applied now
            CandidateIntervention(variable="patch_status", value=0.0),  # not yet applied (delay)
        ),
    )
    mission_trust_loss = {
        PATCH_NOW: abs(min(0.0, trust_loss.results[0].delta_trust)),
        DELAY: abs(min(0.0, trust_loss.results[1].delta_trust)),
    }

    # Algorithm 6, lines 1-2: CyberExposure(a) via real MTES (Section
    # 12.6.1), integrated over the patch-application window (patch_now)
    # and the full remaining communication window (delay) respectively --
    # no placeholder values remain (Phase 6).
    t_patch = 4.0  # patch requires 4 minutes
    t_window = 6.0  # communication window closes in 6 minutes
    exposure = compute_cyber_exposure(
        graph,
        VULNERABLE_FIRMWARE,
        "mission_alpha",
        now=0.0,
        t_patch=t_patch,
        t_window=t_window,
        intrinsic_risk=0.8,  # rho(v): caller-supplied, no SBOM/vuln database implemented yet
        mission_criticality=0.7,  # kappa_m
    )
    cyber_exposure = exposure.as_dict()

    request = PatchSchedulingRequest(
        t_start=12.0,  # mission starts in 12 minutes
        t_patch=t_patch,
        t_window=t_window,
        cyber_exposure=cyber_exposure,
        mission_trust_loss=mission_trust_loss,
    )

    result = pipeline.infer(
        "mission_alpha", now=0.0, node_features=node_features, patch_scheduling_request=request
    )

    print("=== Algorithm 6: AI Patch Scheduling ===")
    print(f"MissionTrustLoss (via Section 10.5 causal engine): {mission_trust_loss}")
    print(f"CyberExposure (via Section 12.6.1 MTES): {cyber_exposure}")
    print(f"  MTES(delay)     window={exposure.delay.window}: {exposure.delay.summary}")
    print(f"  MTES(patch_now) window={exposure.patch_now.window}: {exposure.patch_now.summary}")
    scheduling = result.scheduling_result
    assert scheduling is not None
    print(f"feasible={scheduling.feasible}  cost={scheduling.cost}")
    print(f"selected action: {scheduling.action}")
    print(f"summary: {scheduling.summary}")

    # Section 11 (Cognitive Cyber Reasoning) and Section 12.6.2 (MDRI),
    # both wired into the Explainability Pipeline's existing Evidence- and
    # Graph-levels below (Section 10.7 -- eight levels, never a ninth).
    cognitive_cyber = CognitiveCyberReasoning(memory.episodic)
    consequence = cognitive_cyber.predict(
        FeatureVector(dependency_identity="satcom_fw@3.2"), now=0.0
    )
    dependency_risk = mission_dependency_risk_index(
        graph, "mission_alpha", intrinsic_risk={VULNERABLE_FIRMWARE: 0.8}
    )

    inputs = ExplanationInputs(
        mission_id="mission_alpha",
        trust_posterior=result.posterior.mean,
        uncertainty=result.posterior.variance,
        neighborhood_size=len(graph.k_hop_neighborhood("mission_alpha", 6)),
        hypotheses=result.hypotheses,
        credibility_weights=(
            result.reconciled_decision.credibility_weights if result.reconciled_decision else {}
        ),
        causal_results=trust_loss.results,
        scheduling_result=scheduling,
        cyber_consequence=consequence,
        cyber_exposure=exposure,
        dependency_risk=dependency_risk,
    )
    report = assemble_explanation(inputs)

    print("\n=== Section 10.7: Eight-Level Explainability Pipeline ===")
    for level in report.levels:
        marker = "" if level.available else " (unavailable)"
        print(f"[{level.name}]{marker} {level.content}")


if __name__ == "__main__":
    main()
