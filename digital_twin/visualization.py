"""Visualization (Section 15): mission timeline, trust evolution, cyber
metric evolution, scheduler decisions, and cross-scenario metric
dashboards -- built only from data `digital_twin`/`evaluation` modules
have already computed (Section 15: "Visualizations should reflect
computed data only.").

matplotlib is optional (not a `pyproject.toml` dependency): every function
here imports it lazily and raises the resulting `ImportError` as-is (fail
loudly, with a clear message) if it is not installed, rather than
silently no-op'ing or falling back to a text-only rendering that could be
mistaken for a real plot.

Node-link diagrams of TRMG/attack-graph *structure* (as opposed to the
computed values that structure produces -- trust, MDRI, etc.) are not
implemented here: rendering real graph topology, as Section 15 lists
"TRMG evolution" and "attack graph evolution" among the desired
visualizations, needs a graph-layout library such as networkx, which this
repository does not depend on. What this module visualizes instead is the
*evolution of computed quantities over simulation time* -- trust, the
five other Section 7.1 prediction heads, and MDRI (Section 12.6.2) --
which is the honest, currently-implementable reading of those two items
without adding a new dependency.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

from digital_twin.dataset import MissionRecord
from evaluation.aggregate import ScenarioSummary

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _require_matplotlib() -> ModuleType:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "digital_twin.visualization requires matplotlib, an optional "
            "dependency not installed in this environment "
            "(pip install matplotlib)"
        ) from exc
    return plt


def plot_trust_evolution(
    records: tuple[MissionRecord, ...], *, title: str | None = None
) -> Figure:
    """Trust posterior mean over simulation time, with a +/- 1 stdev band
    (Section 7's Bayesian trust posterior; Section 15's "trust
    evolution")."""
    plt = _require_matplotlib()
    if not records:
        raise ValueError("plot_trust_evolution requires at least one record")

    times = [r.timestamp for r in records]
    means = [r.trust_mean for r in records]
    stdevs = [r.trust_variance**0.5 for r in records]

    fig, ax = plt.subplots()
    ax.plot(times, means, marker="o", label="trust posterior mean")
    ax.fill_between(
        times,
        [m - s for m, s in zip(means, stdevs, strict=True)],
        [m + s for m, s in zip(means, stdevs, strict=True)],
        alpha=0.2,
        label="+/- 1 stdev",
    )
    ax.set_xlabel("simulation time")
    ax.set_ylabel("trust posterior mean")
    ax.set_title(title or f"Trust evolution: {records[0].scenario_name}")
    ax.legend()
    return fig


def plot_mission_timeline(
    records: tuple[MissionRecord, ...], *, title: str | None = None
) -> Figure:
    """Mission phase over time, with cumulative cyber-incident-count
    markers (Section 15's "mission timeline")."""
    plt = _require_matplotlib()
    if not records:
        raise ValueError("plot_mission_timeline requires at least one record")

    times = [r.timestamp for r in records]
    phases = [r.mission_phase for r in records]
    incidents = [r.cyber_incident_count for r in records]

    fig, (ax_phase, ax_incidents) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax_phase.step(times, phases, where="post", marker="o")
    ax_phase.set_ylabel("mission phase")
    ax_phase.set_title(title or f"Mission timeline: {records[0].scenario_name}")

    ax_incidents.step(times, incidents, where="post", marker="o", color="tab:red")
    ax_incidents.set_ylabel("cyber incidents (cumulative)")
    ax_incidents.set_xlabel("simulation time")
    fig.tight_layout()
    return fig


_DASHBOARD_FIELDS: tuple[tuple[str, str], ...] = (
    ("trust_mean", "trust"),
    ("p_success", "P(success)"),
    ("expected_delay", "expected delay"),
    ("cyber_impact", "cyber impact"),
    ("comm_reliability", "comm reliability"),
    ("operational_confidence", "operational confidence"),
    ("mdri_value", "MDRI"),
)


def plot_metric_dashboard(
    records: tuple[MissionRecord, ...], *, title: str | None = None
) -> Figure:
    """A multi-panel dashboard of every Section 7.1 prediction head plus
    MDRI (Section 12.6.2) over simulation time (Section 15's "metric
    dashboards")."""
    plt = _require_matplotlib()
    if not records:
        raise ValueError("plot_metric_dashboard requires at least one record")

    times = [r.timestamp for r in records]
    fig, axes = plt.subplots(
        len(_DASHBOARD_FIELDS), 1, sharex=True, figsize=(8, 2 * len(_DASHBOARD_FIELDS))
    )
    for ax, (field, label) in zip(axes, _DASHBOARD_FIELDS, strict=True):
        values = [getattr(r, field) for r in records]
        ax.plot(times, values, marker="o")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("simulation time")
    fig.suptitle(title or f"Metric dashboard: {records[0].scenario_name}")
    fig.tight_layout()
    return fig


def plot_scheduler_decisions(
    records: tuple[MissionRecord, ...], *, title: str | None = None
) -> Figure:
    """Scheduler action at each query time (Section 15's "scheduler
    decisions"). Records where the scheduler was not invoked (Algorithm 1
    line 11's condition not met, or no scheduler configured) plot as a
    distinct "none" category rather than being omitted, so the absence of
    a decision stays visible instead of silently dropping those rows."""
    plt = _require_matplotlib()
    if not records:
        raise ValueError("plot_scheduler_decisions requires at least one record")

    times = [r.timestamp for r in records]
    actions = [r.scheduler_action or "none" for r in records]

    fig, ax = plt.subplots()
    distinct_actions = sorted(set(actions))
    y_positions = [distinct_actions.index(a) for a in actions]
    ax.scatter(times, y_positions, marker="s")
    ax.set_yticks(range(len(distinct_actions)))
    ax.set_yticklabels(distinct_actions)
    ax.set_xlabel("simulation time")
    ax.set_ylabel("scheduler action")
    ax.set_title(title or f"Scheduler decisions: {records[0].scenario_name}")
    return fig


def plot_scenario_comparison(
    summaries: tuple[ScenarioSummary, ...], metric_field: str, *, title: str | None = None
) -> Figure:
    """Bar chart comparing `metric_field` (one of
    `evaluation.aggregate.FINAL_METRIC_FIELDS`) across scenarios, with
    error bars drawn from each scenario's own confidence interval
    (Section 15's cross-scenario "metric dashboards"). Scenarios whose
    summary has no `final_metric_stats[metric_field]` entry (no run
    produced that metric) are silently excluded from the bars -- an
    honest omission, not a fabricated zero.
    """
    plt = _require_matplotlib()
    available = tuple(s for s in summaries if metric_field in s.final_metric_stats)
    if not available:
        raise ValueError(f"no summary carries final_metric_stats[{metric_field!r}]")

    names = [s.scenario_name for s in available]
    means = [s.final_metric_stats[metric_field].mean for s in available]
    err_low = [
        s.final_metric_stats[metric_field].mean - s.final_metric_stats[metric_field].ci_low
        for s in available
    ]
    err_high = [
        s.final_metric_stats[metric_field].ci_high - s.final_metric_stats[metric_field].mean
        for s in available
    ]

    fig, ax = plt.subplots()
    ax.bar(names, means, yerr=[err_low, err_high], capsize=4)
    ax.set_ylabel(metric_field)
    ax.set_title(title or f"{metric_field} by scenario")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
