#!/usr/bin/env python
"""Example: run Algorithm 1 (TTRN-Infer) end-to-end over a small synthetic TRMG.

This is the *minimal* Algorithm 1 example -- no optional collaborators
configured -- exercising the Temporal Reasoning Memory Graph (Section 6),
Trust Propagation (Section 7.2), the mission-aware GNN's attention and
prediction heads (Section 7.1), the Bayesian posterior update (Section 7.3),
and the TTRN pipeline orchestration (Algorithm 1, Section 5.2). Episodic
Mission Memory, Structured AI Argumentation, Causal Mission Intelligence,
and the AI Patch Scheduler are all implemented elsewhere in this
repository (see `docs/ARCHITECTURE.md`) -- this example simply does not
wire any of them in, by design, to isolate the pipeline's core stages.
See `examples/run_ttrn_pipeline_with_memory.py`,
`..._with_argumentation.py`, `..._with_causal.py`, and
`..._with_scheduler.py` for each one added incrementally, and
`docs/ARCHITECTURE.md` for the full component map.

It deliberately prints the pipeline's `warnings` list too, so it is
obvious at a glance which optional stages this particular call did not
configure, rather than leaving that silent.

Run with:
    python -m examples.run_ttrn_pipeline
"""

from __future__ import annotations

import logging

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from graph.trmg.temporal import TemporalDependencyAnalyzer
from reasoning.trust.criticality import MissionCriticalityModel
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TelemetryEdge, TTRNPipeline


def build_demo_graph() -> TemporalReasoningMemoryGraph:
    """A small hand-built TRMG resembling the manuscript's illustrative
    chain (Section 6.1): an engine-health node feeding a mission node,
    itself depending on SATCOM availability."""
    graph = TemporalReasoningMemoryGraph()
    for node_id in ("engine_health", "satcom", "mission_alpha"):
        graph.add_node(node_id)

    graph.add_edge(
        "engine_health",
        "mission_alpha",
        timestamp=0.0,
        confidence=0.92,
        risk=0.1,
        decay_rate=0.03,
        priority=0.9,
    )
    graph.add_edge(
        "satcom",
        "mission_alpha",
        timestamp=0.0,
        confidence=0.75,
        risk=0.2,
        decay_rate=0.08,
        priority=0.7,
    )
    return graph


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    graph = build_demo_graph()
    dependency_analyzer = TemporalDependencyAnalyzer(graph)
    dependency_analyzer.record_snapshot("mission_alpha", 0.0)

    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    pipeline = TTRNPipeline(graph, gnn, k_hop=6)

    rng = np.random.default_rng(0)
    node_features = {
        "engine_health": rng.normal(size=8),
        "satcom": rng.normal(size=8),
        "mission_alpha": rng.normal(size=8),
    }

    telemetry = (
        TelemetryEdge(
            source="weather",
            target="mission_alpha",
            confidence=0.55,
            decay_rate=0.1,
            priority=0.4,
        ),
    )
    node_features["weather"] = rng.normal(size=8)

    result = pipeline.infer(
        "mission_alpha", now=12.0, node_features=node_features, telemetry=telemetry
    )
    dependency_analyzer.record_snapshot("mission_alpha", 12.0)

    print("\n=== TTRN-Infer(mission_alpha, t=12.0) ===")
    print(f"T_post (trust posterior):  {result.posterior.mean:.4f}")
    print(f"Upsilon (uncertainty):     {result.posterior.variance:.4f}")
    print(f"raw GNN predictions:       {result.predictions}")
    print(f"|N_k(m)| explored:         {len(graph.k_hop_neighborhood('mission_alpha', 6))}")

    volatility = dependency_analyzer.volatility("mission_alpha", 12.0, window=12.0)
    print(f"dependency volatility:    {volatility:.4f}  (Section 6.5)")

    criticality = MissionCriticalityModel(alpha_base=0.5)
    trust_floor = criticality.trust_floor(kappa_m=0.8)
    print(f"trust floor tau(m) (kappa_m=0.8): {trust_floor:.4f}")
    print(f"trust posterior meets floor: {result.posterior.mean >= trust_floor}")

    print("\n--- Optional stages not configured for this example (see below) ---")
    for warning in result.warnings:
        print(f"  * {warning}")


if __name__ == "__main__":
    main()
