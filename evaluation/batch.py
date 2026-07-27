"""Batch execution over the Digital Twin (Section 15) + `TTRNPipeline`
(Section 5), for Section 17's "batch execution, multiple seeds, scenario
sweeps, parameter sweeps" requirement.

Every run drives `digital_twin.integration.run_scenario_with_pipeline`
unchanged -- this module adds no new reasoning path, only the scaffolding
to run that same pipeline many times over a grid of (scenario, seed,
k_hop, embedding_dim) and record what actually happened
(`digital_twin.dataset`), plus wall-clock execution time
(`time.perf_counter`) and peak traced memory (`tracemalloc`, standard
library -- no `psutil` dependency added for this).

By default each run uses a bare `TTRNPipeline` with no memory /
argumentation / causal / scheduler collaborators wired in -- the same
`Null*` defaults `reasoning/ttrn/pipeline.py` itself falls back to.
Section 15.1's scenarios carry no domain-specific vulnerability node,
structural causal model, or patch window that a generic sweep could
supply without fabricating one (compare
`examples/run_ttrn_pipeline_with_scheduler.py`, which hand-builds exactly
that domain context for one specific scenario). Callers who want a richer
sweep -- real Episodic Memory, real Argumentation, a real Structural
Causal Model, a real Patch Scheduler request -- pass their own
`pipeline_factory`.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

from digital_twin.dataset import MissionRecord, build_dataset
from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.scenarios import ScenarioSpec
from digital_twin.twin import MissionDigitalTwin
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline

PipelineFactory = Callable[[MissionDigitalTwin], TTRNPipeline]
ScenarioBuilder = Callable[[], ScenarioSpec]


def default_pipeline_factory(
    twin: MissionDigitalTwin, *, k_hop: int = 6, embedding_dim: int = 16
) -> TTRNPipeline:
    """A bare `TTRNPipeline` over `twin.graph`: real Section 7.1 attention
    + prediction heads, no memory/argumentation/causal/scheduler
    collaborators (see module docstring for why not)."""
    gnn = MissionAwareGNN(embedding_dim=embedding_dim, feature_dim=8, seed=twin.seed)
    return TTRNPipeline(twin.graph, gnn, k_hop=k_hop)


@dataclass(frozen=True)
class RunResult:
    """Everything one scenario run produced: the reproducibility dataset
    (`digital_twin.dataset.MissionRecord`s) plus this run's own execution
    timing and peak memory."""

    scenario_name: str
    seed: int
    k_hop: int
    embedding_dim: int
    execution_time_seconds: float
    peak_memory_bytes: int
    records: tuple[MissionRecord, ...]


def run_single(
    scenario_builder: ScenarioBuilder,
    seed: int,
    *,
    k_hop: int = 6,
    embedding_dim: int = 16,
    pipeline_factory: PipelineFactory | None = None,
) -> RunResult:
    """Run one (scenario, seed, k_hop, embedding_dim) combination to
    completion and return its `RunResult`.

    `tracemalloc` is started and stopped around exactly this run's
    execution, so `peak_memory_bytes` reflects allocations traced during
    this run only -- not a whole-process measurement, and not a claim of
    OS-level resident memory (that would require `psutil`, not a
    dependency of this repository).
    """
    spec = scenario_builder()
    twin = MissionDigitalTwin(spec.name, seed=seed)
    factory = pipeline_factory or (
        lambda t: default_pipeline_factory(t, k_hop=k_hop, embedding_dim=embedding_dim)
    )
    pipeline = factory(twin)

    tracemalloc.start()
    start = time.perf_counter()
    try:
        query_points = run_scenario_with_pipeline(twin, spec, pipeline)
    finally:
        elapsed = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    records = build_dataset(spec.name, seed, query_points)
    return RunResult(
        scenario_name=spec.name,
        seed=seed,
        k_hop=k_hop,
        embedding_dim=embedding_dim,
        execution_time_seconds=elapsed,
        peak_memory_bytes=peak_bytes,
        records=records,
    )


def run_sweep(
    scenario_builders: tuple[ScenarioBuilder, ...],
    seeds: tuple[int, ...],
    *,
    k_hops: tuple[int, ...] = (6,),
    embedding_dims: tuple[int, ...] = (16,),
    pipeline_factory: PipelineFactory | None = None,
) -> tuple[RunResult, ...]:
    """Run every combination of `scenario_builders` x `seeds` x `k_hops` x
    `embedding_dims` (the "scenario sweeps, parameter sweeps" Section 17
    asks for), in that nesting order. Returns one `RunResult` per
    combination -- nothing is skipped or averaged away here; aggregation
    is `evaluation/aggregate.py`'s job.
    """
    results: list[RunResult] = []
    for builder in scenario_builders:
        for seed in seeds:
            for k_hop in k_hops:
                for embedding_dim in embedding_dims:
                    results.append(
                        run_single(
                            builder,
                            seed,
                            k_hop=k_hop,
                            embedding_dim=embedding_dim,
                            pipeline_factory=pipeline_factory,
                        )
                    )
    return tuple(results)
