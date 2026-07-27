#!/usr/bin/env python
"""One-Command Reproducibility Workflow (Phase 8).

Regenerates every dataset, runs every experiment script, runs the
Benchmark Suite (`benchmarks/`) and the No-EMM ablation
(`evaluation/ablation.py`), then exports publication-ready summary tables
(Markdown) and figures (matplotlib-optional) from their real, computed
outputs -- and writes a reproducibility report documenting exactly what
ran, with which seeds, and a content hash of every deterministic result.

WHAT "IDENTICAL OUTPUT ACROSS RUNS" MEANS HERE: every seed-derived
numeric result (scenario ground-truth outcomes, MDRI values, retrieval
precision@1, scheduler-calibration weights, continual-learning task
loss, ...) is fully determined by the fixed seeds below and is hashed for
verification (`reproducibility_report.json`'s `deterministic_content_hash`).
`execution_time_seconds` and `peak_memory_bytes` are real wall-clock/
memory measurements of *this run's own performance*, not a function of
the seed -- they are expected to differ run to run (different hardware,
system load, ...) and are explicitly excluded from that hash rather than
silently included and then blamed on "non-determinism" if they differ.
Optional-dependency behavior (scipy for paired significance tests,
matplotlib for figures) is reported as present/absent in the
reproducibility report, per this phase's brief ("except where documented
optional dependencies introduce acceptable numerical differences").

Run with:
    python -m scripts.reproduce_all
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

from benchmarks.report import BenchmarkSuiteReport, run_benchmark_suite
from digital_twin.dataset import build_dataset
from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.scenarios import ALL_SCENARIO_BUILDERS, build_satellite_blackout
from digital_twin.twin import MissionDigitalTwin
from evaluation.ablation import AblationReport, run_no_emm_ablation
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline
from scripts import experiment_rq6_rq7_memory_retrieval, experiment_rq12_cyber_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets" / "generated"
PUBLICATION_DIR = OUTPUT_DIR / "publication"
SEEDS = tuple(range(10))

_NONDETERMINISTIC_KEYS = {"execution_time_seconds", "peak_memory_bytes"}


def _strip_nondeterministic(value: Any) -> Any:
    """Recursively remove wall-clock/memory measurement fields from a
    (nested dict/list/tuple) structure before hashing -- see module
    docstring. `dataclasses.asdict` preserves tuple-typed fields as
    tuples (not lists), so both container types must be handled or
    tuple-nested fields silently escape stripping.
    """
    if isinstance(value, dict):
        return {
            k: _strip_nondeterministic(v)
            for k, v in value.items()
            if k not in _NONDETERMINISTIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_nondeterministic(v) for v in value]
    return value


def _content_hash(*json_texts: str) -> str:
    """SHA-256 over the deterministic (timing/memory-stripped) content of
    every given JSON document, concatenated in a fixed order."""
    digest = hashlib.sha256()
    for text in json_texts:
        stripped = _strip_nondeterministic(json.loads(text))
        digest.update(json.dumps(stripped, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _write_markdown_tables(
    benchmark_report: BenchmarkSuiteReport, ablation_report: AblationReport, path: Path
) -> None:
    lines: list[str] = ["# Chronos-AEGIS Benchmark & Ablation Summary Tables", ""]

    lines += ["## Scenario Summary (final-query-time metrics)", ""]
    lines += [
        "| Scenario | n | trust_mean | p_success | mdri_dependency_node_count |",
        "|---|---|---|---|---|",
    ]
    for s in benchmark_report.scenario_summaries:
        trust = s.final_metric_stats.get("trust_mean")
        p_succ = s.final_metric_stats.get("p_success")
        dep = s.final_metric_stats.get("mdri_dependency_node_count")
        trust_cell = (
            f"{trust.mean:.4f} [{trust.ci_low:.4f}, {trust.ci_high:.4f}]" if trust else "N/A"
        )
        p_succ_cell = f"{p_succ.mean:.4f}" if p_succ else "N/A"
        dep_cell = f"{dep.mean:.2f}" if dep else "N/A"
        lines.append(
            f"| {s.scenario_name} | {s.n_runs} | {trust_cell} | {p_succ_cell} | {dep_cell} |"
        )

    lines += ["", "## Retrieval Accuracy (RQ6/RQ7)", ""]
    r = benchmark_report.retrieval
    lines += [
        "| n_episodes | precision@1 | majority-class baseline |",
        "|---|---|---|",
        f"| {r.n_episodes} | {r.precision_at_1:.4f} | {r.majority_class_baseline:.4f} |",
    ]

    lines += ["", "## Scheduler Calibration Benchmark", ""]
    sq = benchmark_report.scheduler_quality
    lines += [
        "| initial w_cyber | initial w_trust | final w_cyber | final w_trust | signal strength |",
        "|---|---|---|---|---|",
        f"| {sq.initial_weight_cyber:.4f} | {sq.initial_weight_trust:.4f} | "
        f"{sq.final_weight_cyber:.4f} | {sq.final_weight_trust:.4f} | "
        f"{sq.cyber_signal_strength:.2f} |",
    ]

    lines += ["", "## Continual-Learning Convergence (Section 16, RQ10)", ""]
    lines += [
        "| mission | scenario | task_loss | ewc_penalty | replay_batch_size |",
        "|---|---|---|---|---|",
    ]
    for point in benchmark_report.convergence:
        lines.append(
            f"| {point.mission_index} | {point.scenario_name} | {point.task_loss:.6f} | "
            f"{point.ewc_penalty:.6f} | {point.replay_batch_size} |"
        )

    lines += ["", "## No-EMM Ablation (Section 17.3)", ""]
    lines += [
        f"Condition A: {ablation_report.condition_full}",
        "",
        f"Condition B: {ablation_report.condition_ablated}",
        "",
    ]
    lines += ["| metric | full | no_emm | difference | Cohen's d |", "|---|---|---|---|---|"]
    for comparison in ablation_report.comparisons:
        d = comparison.cohens_d_value
        d_str = f"{d:.4f}" if d is not None else "N/A"
        lines.append(
            f"| {comparison.metric} | {comparison.full_mean:.4f} | {comparison.no_emm_mean:.4f} | "
            f"{comparison.difference:.4f} | {d_str} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_figures(path_dir: Path) -> list[str]:
    """Publication-ready figures (Phase 8), generated only if matplotlib
    is installed (optional dependency; see digital_twin/visualization.py).
    Returns the list of filenames actually written."""
    if importlib.util.find_spec("matplotlib") is None:
        return []

    from digital_twin.visualization import plot_metric_dashboard, plot_trust_evolution

    spec = build_satellite_blackout()
    twin = MissionDigitalTwin(spec.name, seed=0)
    gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
    pipeline = TTRNPipeline(twin.graph, gnn, k_hop=6)
    query_points = run_scenario_with_pipeline(twin, spec, pipeline)
    records = build_dataset(spec.name, twin.seed, query_points)

    written = []
    for name, plot_fn in (
        ("trust_evolution.png", plot_trust_evolution),
        ("metric_dashboard.png", plot_metric_dashboard),
    ):
        fig = plot_fn(records)
        out = path_dir / name
        fig.savefig(out)
        written.append(name)
    return written


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1/4: regenerating experiment datasets ===")
    experiment_rq12_cyber_metrics.main()
    experiment_rq6_rq7_memory_retrieval.main()

    print("\n=== Step 2/4: running Benchmark Suite ===")
    benchmark_report = run_benchmark_suite(ALL_SCENARIO_BUILDERS, seeds=SEEDS)
    benchmark_json = json.dumps(dataclasses.asdict(benchmark_report), indent=2)
    (OUTPUT_DIR / "benchmark_report.json").write_text(benchmark_json, encoding="utf-8")
    print(f"benchmark report written: {OUTPUT_DIR / 'benchmark_report.json'}")

    print("\n=== Step 3/4: running No-EMM ablation ===")
    ablation_report = run_no_emm_ablation(ALL_SCENARIO_BUILDERS, seeds=SEEDS)
    ablation_json = json.dumps(dataclasses.asdict(ablation_report), indent=2)
    (OUTPUT_DIR / "ablation_report.json").write_text(ablation_json, encoding="utf-8")
    print(f"ablation report written: {OUTPUT_DIR / 'ablation_report.json'}")

    print("\n=== Step 4/4: publication assets (tables + figures) ===")
    tables_path = PUBLICATION_DIR / "summary_tables.md"
    _write_markdown_tables(benchmark_report, ablation_report, tables_path)
    print(f"summary tables written: {tables_path}")

    figures = _generate_figures(PUBLICATION_DIR)
    if figures:
        print(f"figures written: {', '.join(figures)}")
    else:
        print("figures skipped: matplotlib is not installed (optional dependency)")

    scipy_available = importlib.util.find_spec("scipy") is not None
    matplotlib_available = importlib.util.find_spec("matplotlib") is not None
    content_hash = _content_hash(benchmark_json, ablation_json)
    reproducibility_report = {
        "seeds": list(SEEDS),
        "scenario_count": len(ALL_SCENARIO_BUILDERS),
        "deterministic_content_hash": content_hash,
        "hash_covers": ["benchmark_report.json", "ablation_report.json"],
        "hash_excludes_fields": sorted(_NONDETERMINISTIC_KEYS),
        "optional_dependencies": {
            "scipy": scipy_available,
            "matplotlib": matplotlib_available,
        },
        "note": (
            "Re-run this script with the same seeds and compare "
            "deterministic_content_hash: it must match exactly regardless "
            "of hardware or timing. execution_time_seconds/peak_memory_bytes "
            "are expected to differ between runs and are excluded from the hash."
        ),
    }
    repro_path = OUTPUT_DIR / "reproducibility_report.json"
    repro_path.write_text(json.dumps(reproducibility_report, indent=2), encoding="utf-8")
    print(f"\nreproducibility report written: {repro_path}")
    print(f"deterministic content hash: {content_hash}")


if __name__ == "__main__":
    main()
