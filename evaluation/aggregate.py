"""Metric aggregation and machine-readable summary export for the
Evaluation Framework (Section 17): "metric aggregation, confidence
intervals where applicable, execution timing, memory usage, trust
evolution, scheduler decisions, cyber metrics. Export machine-readable
summaries."

Aggregates each scenario's own final-query-time metrics (trust, the five
other Section 7.1 prediction heads, and MDRI, Section 12.6.2) across every
run of that scenario in a sweep, plus each run's own execution time and
peak memory (`evaluation.batch.RunResult`). Nothing here computes a metric
`evaluation.batch`/`digital_twin.dataset` did not already compute -- this
module only groups and summarizes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.batch import RunResult
from evaluation.statistics import DescriptiveStats, descriptive_stats

FINAL_METRIC_FIELDS = (
    "trust_mean",
    "p_success",
    "expected_delay",
    "cyber_impact",
    "comm_reliability",
    "operational_confidence",
    "mdri_value",
    "mdri_dependency_node_count",
)
"""The `digital_twin.dataset.MissionRecord` fields aggregated per scenario,
read from each run's *last* record (i.e. its final query time)."""


@dataclass(frozen=True)
class ScenarioSummary:
    """One scenario's aggregated results across every run of it in a
    sweep."""

    scenario_name: str
    n_runs: int
    execution_time_seconds: DescriptiveStats
    peak_memory_bytes: DescriptiveStats
    final_metric_stats: dict[str, DescriptiveStats]
    scheduler_action_counts: dict[str, int]
    warning_free_run_count: int


def _final_metric_value(run: RunResult, field: str) -> float | None:
    if not run.records:
        return None
    value = getattr(run.records[-1], field)
    return None if value is None else float(value)


def summarize_scenario(
    scenario_name: str, runs: tuple[RunResult, ...], *, confidence: float = 0.95
) -> ScenarioSummary:
    """Aggregate every run in `runs` whose `scenario_name` matches."""
    relevant = tuple(r for r in runs if r.scenario_name == scenario_name)
    if not relevant:
        raise ValueError(f"no runs found for scenario {scenario_name!r}")

    final_metric_stats: dict[str, DescriptiveStats] = {}
    for field in FINAL_METRIC_FIELDS:
        values = tuple(
            v for r in relevant if (v := _final_metric_value(r, field)) is not None
        )
        if values:
            final_metric_stats[field] = descriptive_stats(values, confidence=confidence)

    action_counts: dict[str, int] = {}
    warning_free = 0
    for r in relevant:
        if r.records:
            last = r.records[-1]
            if last.scheduler_action is not None:
                action_counts[last.scheduler_action] = (
                    action_counts.get(last.scheduler_action, 0) + 1
                )
            if last.warning_count == 0:
                warning_free += 1

    return ScenarioSummary(
        scenario_name=scenario_name,
        n_runs=len(relevant),
        execution_time_seconds=descriptive_stats(
            tuple(r.execution_time_seconds for r in relevant), confidence=confidence
        ),
        peak_memory_bytes=descriptive_stats(
            tuple(float(r.peak_memory_bytes) for r in relevant), confidence=confidence
        ),
        final_metric_stats=final_metric_stats,
        scheduler_action_counts=action_counts,
        warning_free_run_count=warning_free,
    )


def summarize_sweep(
    runs: tuple[RunResult, ...], *, confidence: float = 0.95
) -> tuple[ScenarioSummary, ...]:
    """Aggregate `runs` (as produced by `evaluation.batch.run_sweep`) into
    one `ScenarioSummary` per distinct `scenario_name`, in sorted order."""
    scenario_names = sorted({r.scenario_name for r in runs})
    return tuple(
        summarize_scenario(name, runs, confidence=confidence) for name in scenario_names
    )


def to_json(summaries: tuple[ScenarioSummary, ...]) -> str:
    return json.dumps([asdict(s) for s in summaries], indent=2)


def write_summary_json(summaries: tuple[ScenarioSummary, ...], path: str | Path) -> None:
    Path(path).write_text(to_json(summaries), encoding="utf-8")
