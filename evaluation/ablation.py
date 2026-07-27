"""Ablation Studies (Phase 8's Publication Assets: "ablation summaries";
manuscript Section 17.3).

Section 17.3 specifies six component-removal ablations. Five of them
(no-causal-reasoning, no-argumentation, no-trust-propagation,
no-cyber-propagation, no-continual-learning) would each need
domain-specific worked scenario data to compare against (see
`examples/run_ttrn_pipeline_with_scheduler.py` for how much
scenario-specific wiring a single *causal* comparison alone requires) --
building all five honestly is out of this phase's scope. This module
implements exactly one, the one every Digital Twin scenario can already
exercise with no additional domain data: **No-EMM** ("disable Episodic
Memory retrieval and update (Algorithms 2-3); Algorithm 1 line 3 and line
15 are skipped").

`run_no_emm_ablation` compares two real, executed conditions over the same
scenarios and seeds -- `full` (a `TTRNPipeline` with a real
`EpisodicMissionMemory` wired in) against `no_emm` (the bare default
pipeline `evaluation.batch.default_pipeline_factory` already produces --
`NullEpisodicMemory`, Algorithm 1 lines 3/15 skipped, exactly matching
that class's own documented behavior) -- and reports the real measured
difference in every final-query-time metric `evaluation.aggregate`
already tracks, plus `memory_episodes_retrieved` and `warning_count`
(both structurally different between the two conditions), via
`evaluation.statistics.cohens_d`. Whatever the measured difference turns
out to be is what is reported -- this module does not assume in advance
that removing memory helps or hurts.

HONEST FINDING (discovered by running this ablation, not assumed in
advance): every Section 7.1 prediction-head metric (`trust_mean`,
`p_success`, etc.) shows *zero* difference between the two conditions.
The reason is architectural, not a flaw in this module: `TTRNPipeline.infer`
(`reasoning/ttrn/pipeline.py`) calls `MissionAwareGNN.embed` without ever
passing `retrieved_memory_features` (that parameter exists on `embed`,
Section 7.1's own "N_k(m) union M_ret" attention pool, but nothing in the
pipeline populates it from the `MemoryRetrievalResult` EMM-Retrieve
already returns) -- so retrieved episodic memory currently has no path
into the GNN's embedding or predictions at all. Only the memory-specific
bookkeeping fields (`memory_episodes_retrieved`, `warning_count`) differ
between conditions. Wiring retrieved episodes into `embed`'s attention
pool would be a genuine architectural change outside Phase 8's task list;
it is documented here as a discovered limitation, not fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

from digital_twin.twin import MissionDigitalTwin
from evaluation.aggregate import FINAL_METRIC_FIELDS
from evaluation.batch import RunResult, ScenarioBuilder, run_single
from evaluation.statistics import cohens_d
from memory.emm import EpisodicMissionMemory
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline

EXTRA_ABLATION_FIELDS: tuple[str, ...] = ("memory_episodes_retrieved", "warning_count")
"""Fields not in `evaluation.aggregate.FINAL_METRIC_FIELDS` but directly
relevant to a memory ablation specifically."""


@dataclass(frozen=True)
class AblationComparison:
    metric: str
    full_mean: float
    no_emm_mean: float
    difference: float
    """full_mean - no_emm_mean."""
    cohens_d_value: float | None
    """None when either sample has fewer than 2 observations (Cohen's d
    is undefined, not fabricated as 0)."""
    n: int


@dataclass(frozen=True)
class AblationReport:
    condition_full: str
    condition_ablated: str
    n_runs_per_condition: int
    comparisons: tuple[AblationComparison, ...]


def _final_value(run: RunResult, field: str) -> float | None:
    if not run.records:
        return None
    value = getattr(run.records[-1], field)
    return None if value is None else float(value)


def run_no_emm_ablation(
    scenario_builders: tuple[ScenarioBuilder, ...],
    seeds: tuple[int, ...],
    *,
    k_hop: int = 6,
    embedding_dim: int = 16,
) -> AblationReport:
    """Run every (scenario, seed) pair under both conditions and report
    the real, measured metric differences."""
    if not scenario_builders:
        raise ValueError("run_no_emm_ablation requires at least one scenario builder")
    if not seeds:
        raise ValueError("run_no_emm_ablation requires at least one seed")

    full_memory = EpisodicMissionMemory()

    def full_factory(twin: MissionDigitalTwin) -> TTRNPipeline:
        gnn = MissionAwareGNN(embedding_dim=embedding_dim, feature_dim=8, seed=twin.seed)
        return TTRNPipeline(twin.graph, gnn, k_hop=k_hop, memory=full_memory)

    full_results: list[RunResult] = []
    no_emm_results: list[RunResult] = []
    for seed in seeds:
        for builder in scenario_builders:
            full_results.append(
                run_single(
                    builder,
                    seed,
                    k_hop=k_hop,
                    embedding_dim=embedding_dim,
                    pipeline_factory=full_factory,
                )
            )
            no_emm_results.append(
                run_single(builder, seed, k_hop=k_hop, embedding_dim=embedding_dim)
            )

    comparisons: list[AblationComparison] = []
    for field in (*FINAL_METRIC_FIELDS, *EXTRA_ABLATION_FIELDS):
        full_values = tuple(
            v for r in full_results if (v := _final_value(r, field)) is not None
        )
        no_emm_values = tuple(
            v for r in no_emm_results if (v := _final_value(r, field)) is not None
        )
        if not full_values or not no_emm_values:
            continue
        full_mean = sum(full_values) / len(full_values)
        no_emm_mean = sum(no_emm_values) / len(no_emm_values)
        d = (
            cohens_d(full_values, no_emm_values)
            if len(full_values) >= 2 and len(no_emm_values) >= 2
            else None
        )
        comparisons.append(
            AblationComparison(
                metric=field,
                full_mean=full_mean,
                no_emm_mean=no_emm_mean,
                difference=full_mean - no_emm_mean,
                cohens_d_value=d,
                n=min(len(full_values), len(no_emm_values)),
            )
        )

    return AblationReport(
        condition_full="full (EpisodicMissionMemory enabled)",
        condition_ablated="no_emm (NullEpisodicMemory -- Algorithm 1 lines 3/15 skipped)",
        n_runs_per_condition=len(full_results),
        comparisons=tuple(comparisons),
    )


__all__ = [
    "AblationComparison",
    "AblationReport",
    "run_no_emm_ablation",
    "EXTRA_ABLATION_FIELDS",
]
