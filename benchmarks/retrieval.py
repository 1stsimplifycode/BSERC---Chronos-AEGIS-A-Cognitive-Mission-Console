"""Retrieval-Accuracy Benchmark (Phase 8's Benchmark Suite; Section 17.5's
"Memory retrieval accuracy: Precision/recall of Algorithm 3's top-k
retrieval against a labeled ground-truth similar-episode set").

Generalizes the leave-one-out precision@1 measurement
`scripts/experiment_rq6_rq7_memory_retrieval.py` introduced (see that
script's module docstring for the full diagnosis of why the *default*
similarity function initially underperformed a majority-class baseline,
and the `memory/features.py` fix that closed most of the gap) into a
reusable function any benchmark run can call, rather than a one-off
script.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.twin import MissionDigitalTwin
from evaluation.batch import ScenarioBuilder
from memory.emm import EpisodicMissionMemory
from memory.episodic import EpisodicEntry
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


@dataclass(frozen=True)
class RetrievalAccuracyResult:
    n_episodes: int
    evaluated: int
    correct: int
    precision_at_1: float
    majority_class_baseline: float
    outcome_distribution: dict[str, int]


def retrieval_precision_at_1(
    scenario_builders: tuple[ScenarioBuilder, ...],
    seeds: tuple[int, ...],
    *,
    embedding_dim: int = 16,
    k_hop: int = 6,
) -> RetrievalAccuracyResult:
    """Run every (scenario, seed) pair with a single shared
    `EpisodicMissionMemory`, then measure leave-one-out precision@1 (does
    the top-ranked *other* episode share the query episode's own
    outcome?) over every stored episode, alongside the majority-class
    baseline for the same episode set.
    """
    shared_memory = EpisodicMissionMemory()
    for seed in seeds:
        for builder in scenario_builders:
            spec = builder()
            twin = MissionDigitalTwin(spec.name, seed=seed)
            gnn = MissionAwareGNN(embedding_dim=embedding_dim, feature_dim=8, seed=seed)
            pipeline = TTRNPipeline(twin.graph, gnn, k_hop=k_hop, memory=shared_memory)
            run_scenario_with_pipeline(twin, spec, pipeline)

    entries = shared_memory.episodic.all_entries()
    n = len(entries)
    if n == 0:
        return RetrievalAccuracyResult(0, 0, 0, float("nan"), float("nan"), {})

    correct = 0
    evaluated = 0
    for entry in entries:

        def _not_self(candidate: EpisodicEntry, _entry: EpisodicEntry = entry) -> bool:
            return candidate is not _entry

        candidates = shared_memory.episodic.search(
            entry.features, 1, entry.time, None, 0.01, predicate=_not_self
        )
        if not candidates:
            continue
        evaluated += 1
        if candidates[0].entry.outcome == entry.outcome:
            correct += 1

    outcome_counts = Counter(e.outcome.value for e in entries)
    majority_baseline = max(outcome_counts.values()) / n

    return RetrievalAccuracyResult(
        n_episodes=n,
        evaluated=evaluated,
        correct=correct,
        precision_at_1=correct / evaluated if evaluated else float("nan"),
        majority_class_baseline=majority_baseline,
        outcome_distribution=dict(outcome_counts),
    )
