"""The Benchmark Suite (Phase 8), built on top of `evaluation/` and
`digital_twin/`.

benchmarks.scalability        TRMG neighborhood (k_hop) and Episodic
                               Memory (|M_e|) scalability (Section 17.4)
benchmarks.convergence        Continual-learning convergence (Section 16,
                               RQ10)
benchmarks.retrieval          Retrieval-accuracy benchmark (RQ6/RQ7)
benchmarks.scheduler_quality  Scheduler-calibration benchmark
benchmarks.report             `run_benchmark_suite`: the one entry point
                               that runs every axis and exports a
                               structured JSON report

See `benchmarks/README.md` for what this package does and does not claim
-- in particular, it does not implement any of Section 17.2's named RQ
baselines (AutoGen/CrewAI-style dialogue, CVSS scoring, etc.); it
benchmarks the real, implemented system against itself.
"""

from benchmarks.convergence import ConvergencePoint, continual_learning_convergence
from benchmarks.report import BenchmarkSuiteReport, run_benchmark_suite, to_json, write_report_json
from benchmarks.retrieval import RetrievalAccuracyResult, retrieval_precision_at_1
from benchmarks.scalability import ScalabilityPoint, k_hop_scalability, memory_size_scalability
from benchmarks.scheduler_quality import SchedulerQualityResult, scheduler_calibration_benchmark

__all__ = [
    "ConvergencePoint",
    "continual_learning_convergence",
    "BenchmarkSuiteReport",
    "run_benchmark_suite",
    "to_json",
    "write_report_json",
    "RetrievalAccuracyResult",
    "retrieval_precision_at_1",
    "ScalabilityPoint",
    "k_hop_scalability",
    "memory_size_scalability",
    "SchedulerQualityResult",
    "scheduler_calibration_benchmark",
]
