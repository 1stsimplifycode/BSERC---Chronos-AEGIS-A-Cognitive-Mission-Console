# Manuscript Consistency Report (Phase 9)

Compares `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf` against the
implementation as it stands after Phase 8. Two kinds of finding are
reported separately: **manuscript-internal** inconsistencies (the
manuscript disagrees with itself) and **implementation** discrepancies
(the code disagrees with the manuscript). Neither kind was silently
corrected; both are documented here.

## 1. Terminology

Spot-checked every core symbol against its code identifier: `sigma(m,t)`
-\> `CognitiveMissionState`, `T_post`/`Upsilon` -\> `BayesianTrustPosterior.mean`/
`.variance`, `kappa_m` -\> `reasoning/trust/criticality.py`, `rho(u->v,t)` -\>
`propagated_trust_contribution`, `Vol(n,t,w)` -\> `graph/trmg/temporal.py`,
`Dep(m)` -\> `cyber.centrality.dependency_nodes`, `w1`/`w2` -\>
`PatchSchedulingRequest.weight_cyber`/`.weight_trust`, `theta`/`theta*`/`F_i`
-\> `reasoning.continual.objective.HeadWeights` /
`ContinualLearningState.reference_heads`/`.fisher`. All consistent.
`reasoning/state.py`'s own docstring documents the general policy: where
the PDF's text extraction strips a Greek glyph, the code uses a
descriptive Python identifier instead of guessing at the stripped symbol,
and records the mapping at the point each symbol is introduced. No
terminology drift found.

**Finding (manuscript-internal, cosmetic)**: Section 5.1's prose calls
TTRN "the nine-stage pipeline," but the same subsection's own Table 1
lists twelve stages (TRMG, Mission-Aware GNN, Bayesian Trust Propagation,
EMM, Multi-Agent Hypothesis Generation, Structured AI Argumentation,
Causal Mission Intelligence, Cognitive Cyber Reasoning, Mission-Aware
Cyber Risk Propagation/Temporal Attack Graph, Mission Scheduler/AI Patch
Scheduler, Mission Digital Twin, Online Continual Learning). This is a
manuscript wording/count mismatch, not an implementation issue -- flagged
for the author to reconcile (either the prose or the table needs
updating), not corrected here since this document does not edit the
manuscript.

## 2. Algorithms

| Algorithm | Manuscript lines | Implementation | Fidelity |
|---|---|---|---|
| Algorithm 1 (TTRN-Infer) | 16 lines, Section 5.2 | `reasoning/ttrn/pipeline.py` `TTRNPipeline.infer` | Line-by-line, documented at each line in the module's own docstring. Table 1's twelve stages map onto Algorithm 1 as: 8 stages are explicit Algorithm 1 lines (TRMG, GNN, Bayesian, EMM, Hypothesis Generation, Argumentation, Scheduler, Causal); the remaining 4 (Cognitive Cyber Reasoning, Cyber Risk Propagation/Attack Graph, Digital Twin, Continual Learning) are *not* named as Algorithm 1 lines in the manuscript either -- they compose with the TRMG from outside Algorithm 1's own control flow (documented in `docs/ARCHITECTURE.md`), which is consistent, not a discrepancy |
| Algorithm 2 (EMM-Update) | Section 8.3 | `memory/emm.py` `EpisodicMissionMemory.update` | Consistent, incl. batched consolidation (line 11) |
| Algorithm 3 (EMM-Retrieve) | Section 8.4 | `memory/emm.py` `.retrieve`, `memory/episodic.py` `.search` | Consistent |
| Algorithm 4 (Structured Argument Aggregation) | Section 9.4 | `agents/argumentation/aggregation.py` | Consistent, incl. line 7's escalation path |
| Algorithm 5 (Causal Counterfactual Generation) | Section 10.5 | `reasoning/causal/algorithm5.py` | Consistent, incl. causal-vs-correlational branch labeling |
| Algorithm 6 (AI Patch Scheduling) | Section 14.1 | `scheduler/algorithm6.py` (lines 5-10) + `cyber/scheduler_integration.py` (lines 1-2) + `reasoning/causal/` (lines 3-4) | Consistent -- lines 1-4 (cost-term computation) are explicitly documented as the caller's responsibility in the manuscript's own text ("given" the two terms); this repository supplies real, non-placeholder implementations for both rather than leaving them to a caller-supplied constant |
| Algorithm 7 (MissionAdapt) | Section 16.5 | `reasoning/continual/algorithm7.py` | Consistent, with one documented scope decision: line 1 (EMM-Update) is satisfied by the caller's own prior `TTRNPipeline.infer` call rather than re-executed, to avoid double-appending the same mission's episode (see that module's own docstring) |

No algorithm's implemented control flow contradicts its manuscript
description. Every deviation from a literal reading (Algorithm 6 lines
1-4's split responsibility; Algorithm 7 line 1's reuse) is explicitly
documented at the point it occurs, not silent.

## 3. Equations vs. executable code

Every formula-bearing module has a corresponding test that checks the
implementation against a hand-computed value, not just "runs without
error": `reasoning/trust/bayesian.py` (precision-weighted fusion vs. hand
arithmetic, `tests/trust/test_bayesian.py`), `cyber/metrics.py` (MTES/MDRI/
CMCS vs. hand-computed graphs, `tests/cyber/test_metrics.py`),
`cyber/centrality.py` (betweenness centrality vs. a hand-worked graph,
`tests/cyber/test_centrality.py`), `scheduler/algorithm6.py` (argmin vs.
hand-computed costs, `tests/scheduler/test_algorithm6.py`),
`reasoning/continual/objective.py` (EWC objective's analytic gradient vs.
finite differences to ~1e-11, `tests/reasoning/continual/test_objective.py`),
`graph/trmg/temporal.py` (Vol(n,t,w) vs. hand-constructed
dependency-change sequences, `tests/graph/test_temporal.py`).

No equation in the manuscript's formalized sections (4, 5, 6, 7, 8, 9, 10,
12.6, 14, 16) lacks a corresponding, tested implementation. Sections whose
equations are explicitly *not* implemented (Section 10.4's causal
discovery; Section 16.4's meta-learning) are documented as future work in
both the manuscript and this repository (see `docs/traceability-matrix.md`).

## 4. Figures

The manuscript's own Appendix B ("Figure Inventory") already classifies
its figures honestly: Figures 1-3 (TTRN architecture, CMI hierarchy, EMM
hierarchy) are rendered schematic diagrams; Figures 4-13 are explicitly
"planned... to accompany the empirical evaluation of Section 17," not
rendered in the manuscript at all. This repository does not contradict
that framing -- it does not claim to have produced any of Figures 4-13.

**Finding (manuscript-internal, cosmetic)**: Appendix B's caption states
"eight are specified here as planned figures," but the table itself lists
ten (Figures 4 through 13). A manuscript wording/count mismatch, not an
implementation issue.

**What this repository can now produce, that the manuscript's planned
list did not anticipate having real data for**: `digital_twin/visualization.py`
(matplotlib-optional) can generate `plot_mission_timeline` and
`plot_trust_evolution`/`plot_metric_dashboard` from genuinely computed
`MissionRecord` data -- these are data-driven analogues of the *spirit*
of planned Figure 8 ("Mission timeline / temporal orchestration") and
Figure 11 ("Continual learning lifecycle"), though not the same diagrams
(the manuscript's planned Fig. 8/11 are conceptual compositions, not time
series of a specific run). The remaining planned figures (4, 5, 6, 7, 9,
10, 12, 13) are graph-topology or architecture-schema diagrams (an
argumentation graph instance, a causal DAG, a TRMG node's history tuple,
...) that would need a graph-layout library (e.g. networkx) this
repository does not depend on -- `digital_twin/visualization.py`'s own
module docstring already documents this scope boundary from Phase 7.
matplotlib is not installed in the audited environment (Phase 9), so no
figure -- planned or data-driven -- was actually rendered during this
audit; see `docs/reproducibility-audit.md` for how to enable it.

## 5. Evaluation methodology vs. available experiments

Section 17.6's statistical analysis plan (>=30 seeds per scenario, 5-fold
cross-validation, paired t-test/Wilcoxon + Holm-Bonferroni, effect sizes)
has not been executed by any experiment in this repository -- every
experiment here uses 10 seeds (`SEEDS = tuple(range(10))`,
`scripts/experiment_rq12_cyber_metrics.py`,
`scripts/experiment_rq6_rq7_memory_retrieval.py`,
`scripts/reproduce_all.py`), not the manuscript's own prescribed minimum
of 30, and none uses 5-fold cross-validation. The statistical *machinery*
Section 17.6 names (paired t-test/Wilcoxon, Holm-Bonferroni, effect
sizes) is implemented and tested (`evaluation/statistics.py`) but not yet
applied at the prescribed scale. This is called out explicitly in
`docs/traceability-matrix.md`'s Section 17.6 row and is not
misrepresented anywhere else in this repository's documentation.

## 6. Assumption documentation consistency

Every place the manuscript underspecifies a formula, constant, or design
choice is marked in the source with a searchable convention. Grep audit
(`grep -rl "ASSUMPTION (documented)"`) found 10 files carrying the
explicit marker: `cyber/__init__.py`, `cyber/attack_graph.py`,
`cyber/cognitive_reasoning.py`, `cyber/consequence.py`, `cyber/metrics.py`,
`cyber/risk_propagation.py`, `evaluation/statistics.py`,
`memory/reliability.py`, `reasoning/continual/objective.py`,
`scheduler/calibration.py`. A broader grep for the related conventions
("documented assumption," "HONEST FINDING," "documented choice") finds 8
additional files using an equivalent phrasing for the same purpose
(e.g. `reasoning/trust/bayesian.py`'s "a deliberate, documented modeling
choice"). No module was found making an unexplained numerical or
structural choice where the manuscript leaves one open -- every instance
checked during this audit had a docstring explaining what was chosen and
why the manuscript does not pin it down further.

## Summary

No implementation contradicts the manuscript. Two manuscript-internal
inconsistencies were found (Section 5.1's "nine-stage" text vs. Table 1's
twelve rows; Appendix B's "eight" vs. its own ten-row table) -- both are
wording/counting slips in the PDF itself, flagged for the author, not
"fixed" by this repository (which does not edit the manuscript). The
implementation's assumptions, algorithm fidelity, and equation-to-code
mapping are consistent and independently testable; the evaluation
methodology gap (10 seeds vs. the manuscript's prescribed >=30, no 5-fold
CV) is real and is already disclosed everywhere results are reported --
see `docs/traceability-matrix.md` and `evaluation/README.md`.
