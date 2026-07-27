"""Scalability Benchmark (Phase 8's Benchmark Suite; Section 17.4's
"Scalability analysis: measuring wall-clock latency and memory footprint
as TRMG node/edge count, Episodic Memory size |M_e|, and agent count |A|
are each scaled independently, against the asymptotic bounds of Section
5.3 and Section 8.5").

Two independent axes are measured here, both against real, executed runs
-- no timing or memory number is estimated or fabricated:

- TRMG neighborhood size, via `k_hop_scalability`, reusing
  `evaluation.batch.run_single`'s existing `k_hop` parameter directly
  (this module does not reimplement it).
- Episodic Memory size |M_e|, via `memory_size_scalability`: a shared
  `EpisodicMissionMemory` is pre-populated with real episodes (each
  produced by actually running a Digital Twin scenario, not synthesized)
  up to each requested size before timing one further query against it --
  the wall-clock/memory cost of a query as |M_e| itself grows, which
  `evaluation.batch`'s own sweep (fresh memory every run) does not
  exercise.

Agent count |A| scaling is not benchmarked here: Section 9.1's nine
domain-specialized agents are fixed by the manuscript (`agents/roster.py`),
and no mechanism in this repository varies |A| at runtime -- benchmarking
a parameter nothing can actually vary would report a fabricated axis.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.scenarios import ScenarioSpec
from digital_twin.twin import MissionDigitalTwin
from evaluation.batch import ScenarioBuilder, run_single
from memory.emm import EpisodicMissionMemory
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


@dataclass(frozen=True)
class ScalabilityPoint:
    """One measured (parameter value -> cost) pair."""

    parameter_name: str
    parameter_value: int
    execution_time_seconds: float
    peak_memory_bytes: int


def k_hop_scalability(
    scenario_builder: ScenarioBuilder, seed: int, k_hops: tuple[int, ...]
) -> tuple[ScalabilityPoint, ...]:
    """Execution time / peak memory as `k_hop` (TRMG neighborhood depth)
    varies, holding the scenario and seed fixed. Reuses
    `evaluation.batch.run_single` exactly -- no separate timing logic."""
    points = []
    for k in k_hops:
        result = run_single(scenario_builder, seed, k_hop=k)
        points.append(
            ScalabilityPoint(
                parameter_name="k_hop",
                parameter_value=k,
                execution_time_seconds=result.execution_time_seconds,
                peak_memory_bytes=result.peak_memory_bytes,
            )
        )
    return tuple(points)


def memory_size_scalability(
    scenario_builder: ScenarioBuilder, seed: int, memory_sizes: tuple[int, ...]
) -> tuple[ScalabilityPoint, ...]:
    """Execution time / peak memory of one scenario run, timed after a
    shared `EpisodicMissionMemory` has genuinely accumulated at least
    `size` episodes, for each `size` in `memory_sizes` (ascending order
    required so growth is monotonic and pre-population is not repeated).
    Every run shares one `EpisodicMissionMemory` -- this is what is being
    scaled, so (unlike `evaluation.batch`) it is not a caller-supplied
    parameter here.
    """
    if list(memory_sizes) != sorted(memory_sizes):
        raise ValueError("memory_sizes must be given in ascending order")

    shared_memory = EpisodicMissionMemory()
    rng = np.random.default_rng(seed)

    def factory(twin: MissionDigitalTwin) -> TTRNPipeline:
        gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=twin.seed)
        return TTRNPipeline(twin.graph, gnn, k_hop=6, memory=shared_memory)

    def _run_one() -> tuple[float, int]:
        twin = MissionDigitalTwin(scenario_builder().name, seed=int(rng.integers(0, 2**31 - 1)))
        pipeline = factory(twin)
        spec: ScenarioSpec = scenario_builder()
        tracemalloc.start()
        start = time.perf_counter()
        try:
            run_scenario_with_pipeline(twin, spec, pipeline)
        finally:
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return elapsed, peak

    points = []
    populated = 0
    for target_size in memory_sizes:
        while populated < target_size:
            _run_one()
            populated += 1
        elapsed, peak = _run_one()
        populated += 1
        points.append(
            ScalabilityPoint(
                parameter_name="memory_size",
                parameter_value=target_size,
                execution_time_seconds=elapsed,
                peak_memory_bytes=peak,
            )
        )
    return tuple(points)
