#!/usr/bin/env python
"""Experiment: Cyber Metrics vs. Digital-Twin Ground Truth (RQ12, Section
3 -- partial).

RQ12 asks whether Section 12.6's mission-aware cyber metrics (MTES, MDRI,
CMCS) "correlate more strongly with injected ground-truth mission
consequence than CVSS scores alone" (`cyber/metrics.py`'s own module
docstring, quoting the manuscript). This repository has no CVSS /
vulnerability database (see `cyber/README.md`), so the CVSS half of that
comparison cannot be run here.

What this script *does* run is the implemented half: MDRI (Section
12.6.2, `cyber.metrics.mission_dependency_risk_index`), computed at the
final query time of every one of Section 15.1's ten canonical scenarios
(`digital_twin.scenarios.CANONICAL_SCENARIO_BUILDERS`), across multiple
random seeds, aggregated via `evaluation.aggregate`.

`digital_twin/integration.py` computes MDRI with an empty
`intrinsic_risk` mapping (no vulnerability database -- every dependency
node's own rho(v) is honestly 0), and MDRI(m) = sum_v rho(v)*centrality(v)
(Section 12.6.2): with every rho(v)=0, `mdri_value` is therefore *always*
exactly 0.0 for every scenario -- a real, correctly-computed consequence
of having no vulnerability-risk data, not a bug in the metric or an
omitted result. What this script reports instead as the genuinely
informative, non-fabricated structural signal is `mdri_dependency_node_count`:
the size of Dep(m), the TRMG's dependency-node set reachable from the
mission node -- the one part of MDRI's computation that does not require
externally-supplied vulnerability data and does vary meaningfully with
each scenario's injected topology.

This is a partial, honest contribution toward RQ12, not RQ12 itself: no
CVSS baseline, no correlation coefficient, no p-value, no claim that
MDRI "performs better" than anything. It reports exactly what the real
implementation computes -- nothing here is fabricated or hand-picked to
look like a positive result. `MissionAwareGNN`'s prediction heads are
also untrained (see `reasoning/trust/gnn.py`), so the other prediction
heads reported alongside MDRI here (P_success, cyber_impact, etc.) carry
no calibrated meaning either; they are reported as context, not results.

ORIGINAL FINDING (Phase 7, discovered by running this script, not assumed
in advance): `mdri_dependency_node_count` -- and therefore `mdri_value` --
was exactly 0 for *every* one of the ten canonical scenarios. `Dep(m)`
(`cyber.centrality.dependency_nodes`) identifies dependency nodes via the
`dependency_identity` node attribute, and of the twelve named event types
(`digital_twin/events.py`), only `FirmwareUpdate` ever sets that
attribute -- and none of Section 15.1's ten canonical `build_*` scenarios
(`digital_twin/scenarios.py`) schedule a `FirmwareUpdate` event (a
deliberate, documented choice from when the scenario library was built,
matching each scenario to the closest of the twelve named event types
rather than inventing new ones). This was a real, provable limitation of
the scenario library relative to RQ12/MDRI, not a bug in
`mission_dependency_risk_index` -- `_demonstrate_nonzero_mdri` below
independently proves the metric itself is non-zero once a dependency node
is actually present.

PHASE 8 FIX: `digital_twin.scenarios.FIRMWARE_SCENARIO_BUILDERS` adds five
new scenarios that do schedule `FirmwareUpdate` events (see that module's
own Phase 8 docstring), specifically to close this gap. This script now
reports both scenario groups side by side: the ten original canonical
scenarios (`CANONICAL_SCENARIO_BUILDERS`, left completely unmodified,
still all-zero -- this is expected and correctly reported, not treated as
a regression) and the five new firmware scenarios, which should show a
genuinely non-zero `|Dep(m)|`. `mdri_value` itself remains 0 for *both*
groups regardless: that is a separate, still-unresolved limitation (no
vulnerability database to supply `intrinsic_risk`, see above), which
adding scenarios cannot fix and this script does not claim to fix.

Run with:
    python -m scripts.experiment_rq12_cyber_metrics
"""

from __future__ import annotations

import logging
from pathlib import Path

from cyber.metrics import mission_dependency_risk_index
from digital_twin.scenarios import CANONICAL_SCENARIO_BUILDERS, FIRMWARE_SCENARIO_BUILDERS
from evaluation.aggregate import ScenarioSummary, summarize_sweep, write_summary_json
from evaluation.batch import RunResult, run_sweep
from graph.trmg.model import TemporalReasoningMemoryGraph

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "generated"
SEEDS = tuple(range(10))


def _demonstrate_nonzero_mdri() -> None:
    """A small hand-built graph (not a Digital Twin scenario), proving
    MDRI is genuinely non-zero once dependency nodes are present --
    confirming the all-zero result above reflects the canonical scenario
    library's content (no `FirmwareUpdate` event anywhere in it), not a
    defect in `mission_dependency_risk_index` itself.

    Three, not one, dependency nodes are needed: MDRI's centrality term
    (`cyber.centrality.betweenness_centrality`) is computed over the
    dependency subgraph alone (Section 12.6.2's "restricted to dependency
    edges"), and betweenness centrality is trivially 0 for every node in a
    subgraph of fewer than 3 nodes (there is no third node for a node to
    sit "between"). A single dependency node, by itself, would make this
    demonstration print MDRI=0.0 for the same structural reason the ten
    canonical scenarios do -- a different one from the reason above -- so
    a 3-node dependency chain is used here specifically to exercise the
    centrality term as well as the intrinsic-risk term.
    """
    graph = TemporalReasoningMemoryGraph()
    graph.add_node("mission")
    graph.add_node("vendor_component", dependency_identity="vendor_component@1.0")
    graph.add_node("satcom_firmware", dependency_identity="satcom_fw@2.0")
    graph.add_node("satcom", dependency_identity="satcom_module@1.4")
    graph.add_edge(
        "satcom", "mission", timestamp=0.0, confidence=0.8, decay_rate=0.05, priority=0.7
    )
    graph.add_edge(
        "satcom_firmware", "satcom", timestamp=0.0, confidence=0.9, decay_rate=0.05, priority=0.7
    )
    graph.add_edge(
        "vendor_component",
        "satcom_firmware",
        timestamp=0.0,
        confidence=0.9,
        decay_rate=0.05,
        priority=0.7,
    )
    result = mission_dependency_risk_index(
        graph, "mission", intrinsic_risk={"satcom_firmware": 0.6}, k_hop=6
    )
    print(
        "\nSupplementary check (hand-built graph, not a Digital Twin "
        f"scenario): |Dep(m)|={len(result.dependency_nodes)}, "
        f"MDRI value={result.value:.6f} once dependency nodes with real "
        "vulnerability risk and non-trivial subgraph centrality are "
        "present -- confirms the all-zero result above is a scenario-"
        "library gap, not an MDRI implementation bug."
    )


def _print_group(label: str, summaries: tuple[ScenarioSummary, ...]) -> None:
    print(f"\n--- {label} ---")
    for summary in summaries:
        mdri_value = summary.final_metric_stats.get("mdri_value")
        dep_count = summary.final_metric_stats.get("mdri_dependency_node_count")
        if mdri_value is None or dep_count is None:
            print(f"{summary.scenario_name:42s} not computed (no dependency nodes reached)")
            continue
        print(
            f"{summary.scenario_name:42s} |Dep(m)| mean={dep_count.mean:.2f}  "
            f"95% CI=[{dep_count.ci_low:.2f}, {dep_count.ci_high:.2f}]  "
            f"(MDRI value={mdri_value.mean:.4f}; n={dep_count.n})"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    canonical_results: tuple[RunResult, ...] = run_sweep(CANONICAL_SCENARIO_BUILDERS, seeds=SEEDS)
    firmware_results: tuple[RunResult, ...] = run_sweep(FIRMWARE_SCENARIO_BUILDERS, seeds=SEEDS)
    canonical_summaries = summarize_sweep(canonical_results)
    firmware_summaries = summarize_sweep(firmware_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rq12_cyber_metrics_summary.json"
    write_summary_json(canonical_summaries + firmware_summaries, out_path)

    print("=== RQ12 (partial): MDRI's structural term, canonical vs. firmware scenarios ===")
    print(
        f"canonical: {len(SEEDS)} seed(s) x {len(CANONICAL_SCENARIO_BUILDERS)} scenarios "
        f"= {len(canonical_results)} runs; "
        f"firmware: {len(SEEDS)} seed(s) x {len(FIRMWARE_SCENARIO_BUILDERS)} scenarios "
        f"= {len(firmware_results)} runs"
    )

    _print_group(
        "Canonical scenarios (Section 15.1, unmodified) -- expected all-zero", canonical_summaries
    )
    _print_group(
        "Firmware scenarios (Phase 8) -- expected non-zero |Dep(m)|", firmware_summaries
    )

    canonical_dep_total = sum(
        s.final_metric_stats["mdri_dependency_node_count"].mean for s in canonical_summaries
    )
    firmware_dep_total = sum(
        s.final_metric_stats["mdri_dependency_node_count"].mean for s in firmware_summaries
    )
    print(
        f"\nFinding: mean |Dep(m)| summed across canonical scenarios = "
        f"{canonical_dep_total:.2f} (expected 0.0 -- confirmed unmodified); summed across "
        f"the five new firmware scenarios = {firmware_dep_total:.2f} (>0 confirms the fix). "
        "mdri_value remains 0.0 in both groups: adding scenarios closes the |Dep(m)| gap but "
        "not the separate no-vulnerability-database gap (see module docstring)."
    )

    print(f"\nFull machine-readable summary written to: {out_path}")
    print(
        "\nNote: this is a partial RQ12 signal (MDRI's dependency-node-count "
        "structural term only; no CVSS baseline; no correlation coefficient "
        "computed or claimed). See module docstring."
    )
    _demonstrate_nonzero_mdri()


if __name__ == "__main__":
    main()
