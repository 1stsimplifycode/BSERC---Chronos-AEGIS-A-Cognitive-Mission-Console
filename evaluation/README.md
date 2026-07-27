# evaluation/ -- Evaluation Framework (Section 17)

**Status: infrastructure implemented; no RQ result is reported anywhere
in this repository.**

The manuscript is explicit that this is a "falsifiable evaluation plan, not
experimental results" (Abstract) and that it reports "a formal architecture
and a falsifiable evaluation plan... not experimental results" as befits the
work's current maturity. This repository maintains that same posture:
nothing under `evaluation/`, `scripts/`, or `benchmarks/` computes or
fabricates a number that would misrepresent the manuscript's actual
empirical status.

## What is implemented

- `evaluation/batch.py`: `run_single` / `run_sweep` -- batch execution of
  the Digital Twin (`digital_twin/`, Section 15) + `TTRNPipeline`
  (Section 5) across scenario x seed x parameter (`k_hop`, `embedding_dim`)
  grids, timed (`time.perf_counter`) and memory-profiled (`tracemalloc`,
  standard library -- no `psutil` dependency added).
- `evaluation/aggregate.py`: `summarize_scenario` / `summarize_sweep` --
  per-scenario aggregation of final-query-time metrics (trust, the other
  five Section 7.1 prediction heads, MDRI), execution time, peak memory,
  scheduler-action counts, and warning-free-run counts, exported as
  machine-readable JSON (`write_summary_json`).
- `evaluation/statistics.py`: `descriptive_stats` (mean, stdev, confidence
  interval), `cohens_d`, `holm_bonferroni` (all standard-library only), and
  `paired_comparison` (paired t-test statistic and Cohen's d always
  computed; the two named p-values -- paired t-test, Wilcoxon signed-rank
  -- are computed only when `scipy` is installed, an optional dependency
  not otherwise used anywhere in this repository; when it is absent both
  p-value fields are `None` with an explanatory `note`, never a fabricated
  or hand-approximated value).
- `digital_twin/visualization.py`: mission timeline, trust evolution,
  metric dashboard, scheduler-decision, and cross-scenario comparison
  plots, built only from already-computed `MissionRecord`/`ScenarioSummary`
  data (matplotlib is optional; see that module's docstring).
- `scripts/experiment_rq12_cyber_metrics.py`,
  `scripts/experiment_rq6_rq7_memory_retrieval.py`: two runnable,
  reproducible experiments demonstrating this infrastructure end to end
  (see "What has actually been run" below).
- `evaluation/ablation.py` (Phase 8): one real Section 17.3 ablation
  (No-EMM), comparing a memory-enabled `TTRNPipeline` against the bare
  default over the same scenarios/seeds.
- `reasoning/continual/` (Phase 8): Section 16's Algorithm 7 and
  Regularized Continual Objective -- see `docs/continual-learning-guide.md`.
- `benchmarks/` (Phase 8): scalability, continual-learning convergence,
  retrieval accuracy, and scheduler-calibration benchmarks -- see
  `docs/benchmark-guide.md`.
- `scripts/reproduce_all.py` (Phase 8): the one-command reproducibility
  workflow -- see `docs/reproducibility-guide.md`.

## What is *not* implemented, and why

Section 17.2's thirteen research questions (RQ1-RQ13, Section 3) each need
a named baseline this repository does not implement: free-form
AutoGen/CrewAI-style dialogue (RQ2), static CVSS scoring and MITRE
ATT&CK-style attack-graph reasoning (RQ9), structural causal discovery
baselines (RQ8), an immediate-patch policy (RQ5), and a CVSS/vulnerability
database (RQ12) -- per Table 0 of the manuscript. None of these baselines
are fabricated here; running a *named RQ as a full comparison* still
requires them. (RQ10's other prerequisite, the continual-learning loop,
Section 16, is now implemented -- `reasoning/continual/` -- but RQ10 as a
full comparison still needs the "static, non-adapting model" baseline
named in Table 2, which is just `TTRNPipeline` with `mission_adapt` never
called, not a separate implementation; no such comparison has been run.)

What changed in Phase 7: the Digital Twin (Section 15) was the other
missing prerequisite this README used to name -- "running any of
RQ1-RQ13 as an actual evaluation still requires the Mission Digital Twin
to generate mission episodes and injected ground-truth outcomes at
scale." That prerequisite now exists, which is why `scripts/` and
`benchmarks/` can run real, partial, non-fabricated RQ-adjacent
experiments even though no RQ's full evaluation -- baseline included --
can run yet.

## What has actually been run

- `scripts/experiment_rq12_cyber_metrics.py` runs MDRI (Section 12.6.2)
  across Section 15.1's ten canonical scenarios x 10 seeds, *and* (Phase
  8) Phase 8's five new firmware scenarios x 10 seeds. **Finding
  (Phase 7):** every canonical scenario yields `mdri_value = 0.0` and
  `|Dep(m)| = 0`, because none of them schedule a `FirmwareUpdate` event
  (the only event type that sets the `dependency_identity` attribute
  `Dep(m)` is keyed on). **Finding (Phase 8):** the five new firmware
  scenarios (`digital_twin.scenarios.FIRMWARE_SCENARIO_BUILDERS`) close
  this gap for real -- every one yields `|Dep(m)| = 1.0`, while the
  original ten remain exactly `0.0` (confirmed unmodified, for
  before/after comparison). `mdri_value` itself remains `0.0` in *both*
  groups regardless -- a separate, still-open limitation (no vulnerability
  database to supply `intrinsic_risk`), which adding scenarios cannot
  fix and this script does not claim to fix. The script also runs a small
  hand-built graph proving MDRI is genuinely non-zero (0.6, in that
  example) once a dependency node with real vulnerability risk and
  non-trivial subgraph centrality is present.
- `scripts/experiment_rq6_rq7_memory_retrieval.py` runs a controlled
  before/after comparison -- on identical stored episodes, not separate
  runs -- of Algorithm 3's real `IndexedSimilaritySearch`
  (`memory.episodic.EpisodicMemoryStore.search`) precision@1 under the
  *default* (unweighted) similarity function. **Finding (Phase 7):** the
  first-ever measurement (13 scenarios x 10 seeds, 130 episodes) found
  precision@1 = 0.3846, *below* the majority-class baseline of 0.6154 --
  default, unweighted similarity retrieval performed worse than always
  guessing the most common outcome. **Fix (Phase 8):**
  `memory.features.extract_feature_vector` now also reads incoming-edge
  confidence into `continuous_features` (see `memory/README.md` for the
  diagnosis). **Finding (Phase 8, current 18-scenario library, 180
  episodes):** BEFORE (categorical fields only) = 0.5556, AFTER
  (+ incoming-edge confidence) = 0.7778, now *above* the 0.5000
  majority-class baseline. This is exactly the kind of manuscript-
  consistent, evidence-driven improvement RQ6/RQ7 calls for -- it is not
  a claim that retrieval is now "solved," and the before/after comparison
  is run fresh every time the script executes, not cached or hand-picked.

Every number above is real, reproducible (fixed seeds), and generated by
the scripts as committed -- re-run them yourself with
`python -m scripts.experiment_rq12_cyber_metrics` /
`python -m scripts.experiment_rq6_rq7_memory_retrieval`, or run
`python -m scripts.reproduce_all` for the full workflow (`docs/reproducibility-guide.md`).
JSON output goes to `datasets/generated/` (gitignored, regenerate rather
than commit).

`MissionAwareGNN`'s prediction heads are untrained (see
`reasoning/trust/gnn.py`'s own module docstring); any trust/P_success/etc.
values these scripts print alongside the findings above are reported as
context only, not as calibrated results.

- `evaluation/ablation.py`'s No-EMM ablation (Phase 8) found that every
  Section 7.1 prediction-head metric is *identical* between the
  memory-enabled and memory-disabled conditions -- because
  `TTRNPipeline.infer` never passes retrieved episodic memory into the
  GNN's embedding at all (see `docs/continual-learning-guide.md`'s
  "Honest findings" section for the full account). Only memory-specific
  bookkeeping (`memory_episodes_retrieved`, `warning_count`) differs.

Section 17.3's other five ablations, 17.4 (testing protocol beyond what's
listed above), 17.5 (expanded metrics beyond trust calibration / ECE,
now in `reasoning/trust/calibration.py`), and 17.7 ("What Would
Constitute a Result") remain unimplemented as evaluation *claims* --
`evaluation.statistics.paired_comparison` and `holm_bonferroni` provide
the statistical machinery Section 17.3/17.6 would use, but no RQ13
superadditivity comparison has been run.

## Testing

`tests/evaluation/`, `tests/benchmarks/`, `tests/reasoning/continual/`,
and `tests/digital_twin/test_visualization.py` cover `batch.py`,
`aggregate.py`, `ablation.py`, `statistics.py`, and every `benchmarks/`
module at or near 100% line coverage (excluding the scipy-gated success
path in `paired_comparison`, an optional dependency not installed in this
environment).
