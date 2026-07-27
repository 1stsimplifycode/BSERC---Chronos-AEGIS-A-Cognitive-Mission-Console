"""Structured Benchmark Reports (Phase 8's Benchmark Suite: "repeated
runs, deterministic seeds, scalability studies, latency, memory usage,
trust evolution, scheduler quality, cyber metrics, retrieval accuracy,
continual-learning convergence. Export structured benchmark reports.").

`run_benchmark_suite` is the one entry point that runs every benchmark
axis and returns a single `BenchmarkSuiteReport`:

- trust evolution + cyber metrics + latency + memory usage: reuses
  `evaluation.batch.run_sweep` / `evaluation.aggregate.summarize_sweep`
  directly (already real, already tested -- not reimplemented here).
- scalability (k_hop, Episodic Memory size): `benchmarks.scalability`.
- continual-learning convergence (Section 16, RQ10): `benchmarks.convergence`.
- retrieval accuracy (RQ6/RQ7): `benchmarks.retrieval`.
- scheduler quality (Phase 8's Scheduler Learning): `benchmarks.scheduler_quality`.

Every field in the resulting report is a real, computed value from an
actual run -- nothing here is estimated, interpolated, or fabricated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.convergence import ConvergencePoint, continual_learning_convergence
from benchmarks.retrieval import RetrievalAccuracyResult, retrieval_precision_at_1
from benchmarks.scalability import ScalabilityPoint, k_hop_scalability, memory_size_scalability
from benchmarks.scheduler_quality import SchedulerQualityResult, scheduler_calibration_benchmark
from evaluation.aggregate import ScenarioSummary, summarize_sweep
from evaluation.batch import ScenarioBuilder, run_sweep


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    """Every benchmark axis Phase 8 asks for, from one suite run."""

    scenario_summaries: tuple[ScenarioSummary, ...]
    """Repeated runs / deterministic seeds / latency / memory usage /
    trust evolution / cyber metrics, via `evaluation.batch` + `evaluation.aggregate`."""
    k_hop_scalability: tuple[ScalabilityPoint, ...]
    memory_size_scalability: tuple[ScalabilityPoint, ...]
    convergence: tuple[ConvergencePoint, ...]
    retrieval: RetrievalAccuracyResult
    scheduler_quality: SchedulerQualityResult


def run_benchmark_suite(
    scenario_builders: tuple[ScenarioBuilder, ...],
    seeds: tuple[int, ...],
    *,
    k_hops: tuple[int, ...] = (3, 6),
    memory_sizes: tuple[int, ...] = (0, 5, 20),
) -> BenchmarkSuiteReport:
    """Run every benchmark axis over `scenario_builders` x `seeds` (plus
    `k_hops` for k-hop scalability and `memory_sizes` for Episodic Memory
    scalability) and return one `BenchmarkSuiteReport`.
    """
    if not scenario_builders:
        raise ValueError("run_benchmark_suite requires at least one scenario builder")
    if not seeds:
        raise ValueError("run_benchmark_suite requires at least one seed")

    results = run_sweep(scenario_builders, seeds)
    scenario_summaries = summarize_sweep(results)

    first_builder = scenario_builders[0]
    first_seed = seeds[0]

    return BenchmarkSuiteReport(
        scenario_summaries=scenario_summaries,
        k_hop_scalability=k_hop_scalability(first_builder, first_seed, k_hops),
        memory_size_scalability=memory_size_scalability(first_builder, first_seed, memory_sizes),
        convergence=continual_learning_convergence(scenario_builders, first_seed),
        retrieval=retrieval_precision_at_1(scenario_builders, seeds),
        scheduler_quality=scheduler_calibration_benchmark(first_seed),
    )


def to_json(report: BenchmarkSuiteReport) -> str:
    return json.dumps(asdict(report), indent=2)


def write_report_json(report: BenchmarkSuiteReport, path: str | Path) -> None:
    Path(path).write_text(to_json(report), encoding="utf-8")
