#!/usr/bin/env python
"""Example: Episodic Mission Memory (Section 8) in action across three
missions sharing the same vulnerable-dependency/topology pattern.

Demonstrates, with the real `EpisodicMissionMemory` (memory/emm.py) wired
into `TTRNPipeline` instead of the default `NullEpisodicMemory`:

  1. EMM-Update (Algorithm 2) persisting a concluded mission's episode.
  2. EMM-Retrieve (Algorithm 3) finding that episode on a later, similar
     mission -- populating sigma(m,t)'s M_ret coordinate for real.
  3. Batched consolidation (Section 8.3.1) distilling a semantic rule once
     the same (dependency, topology) pattern has recurred with a
     consistent outcome across enough missions.

Run with:
    python -m examples.run_ttrn_pipeline_with_memory
"""

from __future__ import annotations

import logging

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.consolidation import ConsolidationConfig
from memory.emm import EpisodicMissionMemory
from memory.outcome import MissionOutcome
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


def build_graph() -> TemporalReasoningMemoryGraph:
    graph = TemporalReasoningMemoryGraph()
    graph.add_node(
        "engine_health", dependency_identity="pump_controller_fw@2.1", topology_class="star"
    )
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # theta_pattern=3, interval_missions=3: consolidation (and the semantic
    # rule it can distill) becomes due exactly on the third concluded mission.
    memory = EpisodicMissionMemory(
        consolidation_config=ConsolidationConfig(
            interval_missions=3, retention_bound=100, theta_pattern=3
        )
    )
    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    rng = np.random.default_rng(0)

    for i in range(3):
        graph = build_graph()  # a fresh TRMG per mission, same dependency/topology pattern
        pipeline = TTRNPipeline(graph, gnn, k_hop=6, memory=memory)
        node_features = {
            "engine_health": rng.normal(size=8),
            "mission_alpha": rng.normal(size=8),
        }

        result = pipeline.infer(
            "mission_alpha",
            now=float(i),
            node_features=node_features,
            outcome=MissionOutcome.FAILURE,  # every mission in this pattern fails
        )
        retrieved_count = len(result.state.retrieved_memory)
        print(
            f"mission {i}: T_post={result.posterior.mean:.4f}  "
            f"retrieved={retrieved_count} prior episode(s)"
        )

    print(f"\nepisodic entries stored: {len(memory.episodic)}")
    print(f"consolidation ran: {memory.last_consolidation is not None}")
    if memory.last_consolidation:
        for rule in memory.last_consolidation.updated_rules:
            print(
                f"  distilled semantic rule: (dependency={rule.dependency_identity}, "
                f"topology={rule.topology_class}) -> {rule.predicted_outcome.value} "
                f"(support={rule.support_count})"
            )

    # A fourth, brand-new mission with the same pattern can now retrieve
    # both the prior episodes AND the distilled semantic rule.
    graph = build_graph()
    pipeline = TTRNPipeline(graph, gnn, k_hop=6, memory=memory)
    node_features = {"engine_health": rng.normal(size=8), "mission_alpha": rng.normal(size=8)}
    result = pipeline.infer("mission_alpha", now=3.0, node_features=node_features)

    retrieved_count = len(result.state.retrieved_memory)
    print(f"\nmission 3 (query only): retrieved {retrieved_count} prior episode(s)")
    print(
        "reconciled_decision coordinate: still None -- no hypothesis_generators/argumentation "
        "wired into this example's pipeline (see examples/run_ttrn_pipeline_with_argumentation.py "
        "for Section 9 in action)"
    )


if __name__ == "__main__":
    main()
