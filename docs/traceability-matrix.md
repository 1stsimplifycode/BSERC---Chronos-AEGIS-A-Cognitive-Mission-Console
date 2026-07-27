# Claim-to-Evidence Traceability Matrix (Phase 9)

Source of truth: `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf`. This matrix
maps every major claim in the manuscript to its implementation,
experiment(s), figure(s), table(s), test(s), and source location, and
classifies it honestly. It does not upgrade a claim's status because the
surrounding architecture is real; each row is judged on its own evidence.

## Classification legend

- **Validated** -- implemented, and an executed experiment/test produces
  real evidence supporting the specific claim (not just "the code runs").
- **Implemented, not validated** -- the mechanism is real and tested for
  *correctness* (unit tests, worked examples), but the manuscript's
  empirical claim about it (accuracy, superiority to a baseline,
  calibration quality, ...) has not been experimentally measured.
- **Future work** -- the manuscript itself labels the item as not solved
  / a design possibility / left to future work. Its absence here is
  consistent with the manuscript, not a gap.
- **Unsupported** -- the manuscript states or implies something this
  repository does not (yet) evidence at all.

Manuscript section numbers refer to the numbering used throughout this
document (matches the PDF's own section numbers).

## Section 4 -- The Cognitive Mission Intelligence Framework

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 4.1 | Cognitive Mission State sigma(m,t) is a well-defined tuple threading through every component | `reasoning/state.py` `CognitiveMissionState` | -- | -- | -- | `tests/reasoning/test_state.py` | `reasoning/state.py` | Validated (structural: every field is populated by a real pipeline stage, verified end-to-end in `tests/reasoning/test_ttrn_pipeline.py`) |
| 4.2 | Mission Utility U(sigma) unifies T, P_success, D, C, Sigma into one optimization objective | `reasoning/utility.py` | -- | -- | -- | `tests/reasoning/test_utility.py` | `reasoning/utility.py` | Validated (formula computed correctly against hand-worked values; not used to drive an actual optimization/decision anywhere) |

## Section 5 -- TTRN: CMI's Core Reasoning Engine

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 5.2 | Algorithm 1 (TTRN-Infer) -- the 16-line control flow -- is fully realizable | `reasoning/ttrn/pipeline.py` `TTRNPipeline.infer` | `examples/run_ttrn_pipeline*.py` (5 scripts, incremental to full capstone); every Digital Twin scenario run | -- | -- | `tests/reasoning/test_ttrn_pipeline.py` | `reasoning/ttrn/pipeline.py` | Validated -- every line has real control flow; `Null*` components make an unconfigured stage an honest no-op/warning, never fabricated output |
| 5.3 | Complexity bounds (O(log\|M_e\|) memory, O(k\|N_k(m)\|) attention, etc.) | Implied by the data structures used (indexed `EpisodicMemoryStore`, bounded k-hop traversal) | `benchmarks.scalability` measures real wall-clock/memory scaling | -- | `benchmark_report.json`'s `k_hop_scalability`/`memory_size_scalability` | -- | `graph/trmg/model.py`, `memory/episodic.py`, `benchmarks/scalability.py` | Implemented, not validated -- real timing/memory numbers exist, but no regression fit against the claimed asymptotic bounds has been performed |
| 5.4 | Proposition 3: Bayesian posterior uncertainty is non-increasing under genuine new evidence | `reasoning/trust/bayesian.py` `bayesian_trust_update` | -- | -- | -- | `tests/trust/test_bayesian.py` (explicit monotonic-decrease assertions) | `reasoning/trust/bayesian.py` | Validated |
| 5.4 | Proposition 4: history logs are append-only (TRMG node history, Episodic Memory) | `graph/trmg/model.py` `NodeHistory` (no `update_*`/`remove_*` methods by construction); `memory/episodic.py` `EpisodicMemoryStore.append` | -- | -- | -- | `tests/graph/test_trmg.py`, `tests/memory/test_episodic.py` | `graph/trmg/model.py`, `memory/episodic.py` | Validated (enforced by the type/API, not just convention) |

## Section 6 -- The Temporal Trust Graph / TRMG

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 6.1 | TTG: every edge carries chi(e) = (t, w, c, r, lambda, d, pi) | `graph/trmg/model.py` `EdgeAttributes` | -- | -- | -- | `tests/graph/test_trmg.py` | `graph/trmg/model.py` | Validated |
| 6.2 | TRMG extends TTG with a six-log history tuple rho(n) per node | `graph/trmg/model.py` `NodeHistory` (six append-only logs) | -- | -- | -- | `tests/graph/test_trmg.py` | `graph/trmg/model.py` | Validated |
| 6.5 | Temporal Dependency Analysis: volatility score Vol(n,t,w) | `graph/trmg/temporal.py` | -- | -- | -- | `tests/graph/test_temporal.py` | `graph/trmg/temporal.py` | Validated (formula verified against hand-computed dependency-change sequences); *not* independently confirmed as "predictive of subsequent mission-trust degradation" (Section 17.4's own proposed test for that claim has not been run) |
| (Phase 8) | TRMG supports historical snapshots and reversible temporal pruning | `graph/trmg/snapshot.py`, `TemporalReasoningMemoryGraph.snapshot`/`.restore`/`.prune_history`, `NodeHistory.prune` | -- | -- | -- | `tests/graph/test_snapshot.py` | `graph/trmg/model.py`, `graph/trmg/snapshot.py` | Validated -- not a manuscript claim per se (Phase 8's own "TRMG Evolution" requirement), included for completeness |

## Section 7 -- Mission-Aware GNN, Trust Propagation, Bayesian Update, Criticality

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 7.1 | Mission-aware GNN attention + six coupled prediction heads (T, P_success, D, C, R_comm, O) | `reasoning/trust/gnn.py` `MissionAwareGNN` | `benchmarks.convergence` runs it over real mission sequences | -- | `benchmark_report.json`'s `convergence` | `tests/trust/test_gnn.py` | `reasoning/trust/gnn.py` | Implemented, not validated for prediction quality -- **the module's own docstring states this explicitly**: weights are randomly initialized, no training data/protocol has been run. The *architecture* (attention weights, shapes, ranges) is validated |
| 7.2 | Trust Propagation rho(u-\>v,t): priority/decay-weighted incoming-edge contribution | `reasoning/trust/propagation.py` | -- | -- | -- | `tests/trust/test_propagation.py` | `reasoning/trust/propagation.py` | Validated |
| 7.3 | Bayesian posterior update (precision-weighted fusion) | `reasoning/trust/bayesian.py` | -- | -- | -- | `tests/trust/test_bayesian.py` | `reasoning/trust/bayesian.py` | Validated (documented modeling choice: 1-D Gaussian precision-weighted fusion, not manuscript-pinned to a specific likelihood family) |
| 7.4 | Mission Criticality Model kappa_m | `reasoning/trust/criticality.py` | -- | -- | -- | `tests/trust/test_criticality.py` | `reasoning/trust/criticality.py` | Validated |

## Section 8 -- Episodic Mission Memory

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 8.2 | Four memory tiers M_w, M_s, M_e, M_m | `memory/working.py`, `memory/semantic.py`, `memory/episodic.py`, `memory/meta.py` | `examples/run_ttrn_pipeline_with_memory.py` | -- | -- | `tests/memory/test_*` | `memory/emm.py` (composition) | Validated |
| 8.3 | Algorithm 2 (EMM-Update) incl. batched consolidation | `memory/emm.py`, `memory/consolidation.py` | -- | -- | -- | `tests/memory/test_emm.py`, `tests/memory/test_consolidation.py` | `memory/emm.py` | Validated |
| 8.4 | Algorithm 3 (EMM-Retrieve), IndexedSimilaritySearch, sim(q,s_i) | `memory/episodic.py` `EpisodicMemoryStore.search`, `memory/similarity.py` | `scripts/experiment_rq6_rq7_memory_retrieval.py` | -- | `rq6_rq7_memory_retrieval_precision.json` | `tests/memory/test_episodic.py`, `tests/memory/test_similarity.py` | `memory/episodic.py` | Validated for correctness; retrieval-*quality* (precision@1) is separately, honestly measured -- see 8.6 row below (not automatically "good" just because it's tested) |
| 8.5 | Complexity: O(log\|M_e\|) retrieval under an indexed structure | `memory/episodic.py`'s hash-bucket index (documented as a simplified index, not a full ANN structure) | `benchmarks.scalability.memory_size_scalability` | -- | `benchmark_report.json` | -- | `memory/episodic.py`, `benchmarks/scalability.py` | Implemented, not validated -- real timing exists; no formal complexity-class confirmation |
| 8.6 | Similarity weights and retention bound are open empirical questions (RQ6, RQ7) | `memory/similarity.py` `SimilarityWeights` (unweighted default); `memory/episodic.py` `prune_oldest` | `scripts/experiment_rq6_rq7_memory_retrieval.py` | -- | `rq6_rq7_memory_retrieval_precision.json` | `tests/memory/test_similarity.py` | `memory/similarity.py` | Validated **as an open question** -- the manuscript's own framing (not yet calibrated) is upheld; Phase 8 measured a real, reproducible precision@1 (0.5556 -\> 0.7778 after a feature-coverage fix) but did not calibrate the weights themselves via any learning procedure |
| (Phase 8) | Episodic Learning: outcome-driven, non-destructive retrieval-confidence adjustment | `memory/reliability.py` `ReliabilityLedger` | -- | -- | -- | `tests/memory/test_reliability.py`, `tests/memory/test_emm.py` | `memory/reliability.py` | Validated -- not itself a manuscript claim (Phase 8's own Episodic Learning requirement), included for completeness |

## Section 9 -- Structured AI Argumentation

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 9.1 | Nine domain-specialized agents | `agents/roster.py`, `agents/hypothesis.py` | -- | -- | -- | `tests/agents/test_hypothesis.py` | `agents/roster.py` | Validated (roster + hypothesis structure only; no agent has real domain-specific reasoning logic beyond the structured hypothesis it's handed -- see Section 9.1's own row in `docs/ARCHITECTURE.md`) |
| 9.2 | Argumentation Graph AG = (H, Attacks), Dung-style extension | `agents/argumentation/graph.py` | -- | -- | -- | `tests/agents/argumentation/test_graph.py` | `agents/argumentation/graph.py` | Validated |
| 9.3 | Consensus, Conflict(H), agent credibility weights | `agents/argumentation/consensus.py`, `memory/meta.py` | -- | -- | -- | `tests/agents/argumentation/test_consensus.py`, `tests/memory/test_meta.py` | `agents/argumentation/consensus.py` | Validated |
| 9.4 | Algorithm 4 (Structured Argument Aggregation), incl. escalation | `agents/argumentation/aggregation.py` | `examples/run_ttrn_pipeline_with_argumentation.py` (incl. an escalation case) | -- | -- | `tests/agents/argumentation/test_aggregation.py`, `test_semantics.py` | `agents/argumentation/aggregation.py` | Validated |
| 9.7 | The Explainable Decision Engine | `reasoning/causal/explainability.py` (Section 10.7 implements this; see that section) | `examples/run_ttrn_pipeline_with_causal.py`, `..._with_scheduler.py` | -- | -- | `tests/reasoning/causal/test_explainability.py` | `reasoning/causal/explainability.py` | Validated |

## Section 10 -- Causal Mission Intelligence

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 10.1 | Retained correlational Counterfactual Engine | `reasoning/causal/counterfactual.py` | -- | -- | -- | `tests/reasoning/causal/test_counterfactual.py` | `reasoning/causal/counterfactual.py` | Validated |
| 10.2 | Structural Causal Model (domain-supplied structural functions) | `reasoning/causal/scm.py` | `examples/run_ttrn_pipeline_with_scheduler.py`'s worked SCM | -- | -- | `tests/reasoning/causal/test_scm.py` | `reasoning/causal/scm.py` | Validated |
| 10.3 | Do-calculus / correlation-causation distinction, worked queries | `reasoning/causal/algorithm5.py` | Worked queries in module docstrings + examples | -- | -- | `tests/reasoning/causal/test_algorithm5.py` | `reasoning/causal/algorithm5.py` | Validated |
| 10.4 | Causal discovery is an open problem, not solved | Not implemented -- SCM is domain-supplied, never learned | -- | -- | -- | -- | `reasoning/causal/scm.py`'s own docstring | Future work (manuscript's own classification; consistent) |
| 10.5 | Algorithm 5 (Causal Counterfactual Generation), causal-vs-correlational labeling | `reasoning/causal/algorithm5.py` `StructuralCausalEngine` | `examples/run_ttrn_pipeline_with_causal.py`, `..._with_scheduler.py` | -- | -- | `tests/reasoning/causal/test_algorithm5.py` | `reasoning/causal/algorithm5.py` | Validated |
| 10.7 | Eight-Level Explainability Pipeline | `reasoning/causal/explainability.py` | `examples/run_ttrn_pipeline_with_causal.py` prints all eight levels | -- | -- | `tests/reasoning/causal/test_explainability.py` | `reasoning/causal/explainability.py` | Validated |

## Section 11-13 -- Cognitive Cyber Reasoning, Risk Propagation, Cyber Metrics, Attack Graphs

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 11 | Cognitive Cyber Reasoning (mu(q) mission-consequence prediction) | `cyber/cognitive_reasoning.py`, `cyber/consequence.py` | -- | -- | -- | `tests/cyber/test_cognitive_reasoning.py`, `test_consequence.py` | `cyber/cognitive_reasoning.py` | Validated |
| 12.1-12.2 | Base risk propagation (widest-path) + AI-driven attack-chain reasoning | `cyber/risk_propagation.py`, `cyber/attack_chain.py` | -- | -- | -- | `tests/cyber/test_risk_propagation.py`, `test_attack_chain.py` | `cyber/risk_propagation.py` | Validated |
| 12.3 | Supply-chain / firmware trust reasoning | `cyber/risk_propagation.py` `refresh_dependency_trust` | `digital_twin` firmware scenarios (Phase 8) exercise this for real | -- | `rq12_cyber_metrics_summary.json` | `tests/cyber/test_risk_propagation.py`, `tests/digital_twin/test_events.py` (`FirmwareUpdate`, `PatchCompleted`) | `cyber/risk_propagation.py` | Validated |
| 12.4-12.5 | Zero-trust reasoning, adversarial AI robustness (4 mitigations) | `cyber/robustness.py` | -- | -- | -- | `tests/cyber/test_robustness.py` | `cyber/robustness.py` | Implemented, not fully validated -- two of the four named mitigations reuse Sections 6/9 mechanisms directly (documented) rather than new, separately-tested logic |
| 12.6.1 | MTES: rho(v) x attenuation x psi x time-integral | `cyber/metrics.py` `mission_trust_exposure_score` | `cyber/scheduler_integration.py` feeds it into Algorithm 6 for real | -- | -- | `tests/cyber/test_metrics.py`, `test_scheduler_integration.py` | `cyber/metrics.py` | Validated for correctness; **not validated as "more predictive than CVSS"** (RQ12/RQ3's actual claim) -- no CVSS baseline exists |
| 12.6.2 | MDRI: sum_v rho(v)\*centrality(v) | `cyber/metrics.py` `mission_dependency_risk_index`, `cyber/centrality.py` | `scripts/experiment_rq12_cyber_metrics.py` | -- | `rq12_cyber_metrics_summary.json`, `benchmark_report.json` | `tests/cyber/test_metrics.py`, `test_centrality.py` | `cyber/metrics.py` | Validated for correctness (incl. a hand-built non-zero proof); **the metric's own numeric value is 0.0 for every Digital Twin scenario run to date** (no vulnerability database) -- see `docs/negative-results.md` |
| 12.6.3 | CMCS: MTES x (1 + normalized centrality) x P_success^-1 | `cyber/metrics.py` `cyber_mission_criticality_score` | -- | -- | -- | `tests/cyber/test_metrics.py` | `cyber/metrics.py` | Validated for correctness; same "no CVSS baseline, no vulnerability database" caveat as MTES/MDRI |
| 13 | Temporal Attack Graphs, integrated into the TRMG | `cyber/attack_graph.py` | -- | -- | -- | `tests/cyber/test_attack_graph.py` | `cyber/attack_graph.py` | Validated (documented deviation: integrated into TRMG rather than a separate graph structure) |

## Section 14 -- AI Patch Scheduler

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 14.1 | Algorithm 6: feasibility check + weighted-cost argmin over {patch_now, delay} | `scheduler/algorithm6.py` | `examples/run_ttrn_pipeline_with_scheduler.py` reproduces the manuscript's own worked scenario | -- | -- | `tests/scheduler/test_algorithm6.py`, `tests/cyber/test_pipeline_integration.py` | `scheduler/algorithm6.py` | Validated |
| 14.1 | CyberExposure(a) computed via MTES (not a placeholder) | `cyber/scheduler_integration.py` `compute_cyber_exposure` | `examples/run_ttrn_pipeline_with_scheduler.py` | -- | -- | `tests/cyber/test_scheduler_integration.py` | `cyber/scheduler_integration.py` | Validated |
| (Phase 8) | Scheduler Learning: evidence-driven w_cyber/w_trust calibration, Algorithm 6 unmodified | `scheduler/calibration.py` `calibrate_weights` | `benchmarks.scheduler_quality.scheduler_calibration_benchmark` | -- | `benchmark_report.json` | `tests/scheduler/test_calibration.py` | `scheduler/calibration.py` | Validated -- own documented mechanism (Section 16 does not name the scheduler among its covered parameters), not a manuscript-specified formula |

## Section 15 -- Mission Digital Twin

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 15 | Digital Twin as shared simulation substrate; source of ground-truth outcomes | `digital_twin/twin.py`, `digital_twin/integration.py` | Every experiment/benchmark in this repository | -- | Every `datasets/generated/*.json` | `tests/digital_twin/test_twin.py`, `test_integration.py` | `digital_twin/twin.py` | Validated |
| 15.1 | Ten named canonical scenarios | `digital_twin/scenarios.py` `CANONICAL_SCENARIO_BUILDERS` | `scripts/experiment_rq12_cyber_metrics.py`, all benchmarks | -- | `rq12_cyber_metrics_summary.json` | `tests/digital_twin/test_scenarios.py` | `digital_twin/scenarios.py` | Validated |
| 15 | Controlled intervention as substrate for causal discovery (Section 10.4) | Scenarios inject known perturbations with known ground truth | -- | -- | -- | `tests/digital_twin/test_scenarios.py` | `digital_twin/scenarios.py` | Implemented, not validated -- interventions exist and are controlled, but no causal-discovery *algorithm* consumes them (Section 10.4 remains future work) |

## Section 16 -- Adaptive Learning and Continual Reasoning

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 16.1 | Explicit online continual-learning loop (outcome -\> memory -\> TRMG -\> credibility -\> trust -\> improved reasoning) | `reasoning/continual/algorithm7.py` `mission_adapt` | `benchmarks.convergence.continual_learning_convergence` | -- | `benchmark_report.json`'s `convergence` | `tests/reasoning/continual/test_algorithm7.py` | `reasoning/continual/algorithm7.py` | Validated (the loop executes; see 16.2/16.6 rows for what "improved reasoning" specifically does and does not mean here) |
| 16.2 | Regularized Continual Objective L = L_task + lambda\*Sum F_i(theta_i-theta*_i)^2 | `reasoning/continual/objective.py` | Analytic gradient verified against finite differences | -- | -- | `tests/reasoning/continual/test_objective.py` | `reasoning/continual/objective.py` | Validated for correctness of the formula and its gradient; **the penalty term's gradient is structurally zero in every measured run** under the literal single-step-per-mission reading of Algorithm 7 -- see `docs/negative-results.md` |
| 16.3 | Experience replay batch mixed with mission_k | `reasoning/continual/algorithm7.py` (draws from `memory.episodic`) | `benchmarks.convergence` | -- | `benchmark_report.json` | `tests/reasoning/continual/test_algorithm7.py` | `reasoning/continual/algorithm7.py` | Validated |
| 16.4 | Meta-learning outer loop -- "a design possibility, not a claim" | Not implemented | -- | -- | -- | -- | manuscript Section 16.4 | Future work (manuscript's own classification; correctly not implemented) |
| 16.5 | Algorithm 7 (MissionAdapt), all 8 lines | `reasoning/continual/algorithm7.py` | `benchmarks.convergence` | -- | `benchmark_report.json` | `tests/reasoning/continual/test_algorithm7.py` | `reasoning/continual/algorithm7.py` | Validated (line 1 satisfied by the caller's own prior `TTRNPipeline.infer` call, documented, not duplicated) |
| 16.6 | "We have not evaluated whether lambda and replay composition prevent catastrophic forgetting in practice" | N/A -- the manuscript's own admission | `benchmarks.convergence` measures task loss trajectory but does not test forgetting | -- | -- | -- | manuscript Section 16.6 | Unresolved by design -- correctly *not* claimed as solved anywhere in this repository; this is the literal open question RQ10 names |

## Section 17 -- Evaluation Protocol

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 17.2 RQ1 | TRMG vs. static/non-trust-aware graphs | No static-graph or non-trust-aware-GNN baseline implemented | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ2 | Structured argumentation vs. AutoGen/CrewAI-style dialogue | No dialogue baseline implemented | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ3 | Cyber propagation vs. static CVSS | No CVSS baseline / vulnerability database | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ4 | Causal/counterfactual vs. unexplained; human-subjects trust/decision-speed | No human-subjects study | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ5 | AI patch scheduler vs. patch-immediately policy | Scheduler implemented; "patch-immediately" baseline not implemented as a comparison run | -- | -- | -- | `tests/scheduler/test_algorithm6.py` (scheduler correctness only) | `scheduler/algorithm6.py` | Implemented, not validated |
| 17.2 RQ6 | Episodic memory vs. memoryless TTRN | Both conditions exist (`NullEpisodicMemory` vs. `EpisodicMissionMemory`) | `evaluation/ablation.py`'s No-EMM ablation | -- | `ablation_report.json` | `tests/evaluation/test_ablation.py` | `evaluation/ablation.py` | Validated as a real, executed comparison -- **finding: zero difference on every GNN prediction-head metric** (see `docs/negative-results.md`), a genuine, not-yet-favorable result for this RQ |
| 17.2 RQ7 | Tiered memory vs. undifferentiated (flat) memory | No flat-memory baseline implemented; only the tiered design exists | `scripts/experiment_rq6_rq7_memory_retrieval.py` measures retrieval precision (not a tiered-vs-flat comparison) | -- | `rq6_rq7_memory_retrieval_precision.json` | `tests/memory/test_episodic.py` | `memory/episodic.py` | Implemented, not validated (retrieval precision measured; the *comparison this RQ actually names* has not been run) |
| 17.2 RQ8 | Causal vs. correlational reasoning: explanation fidelity, trust-prediction accuracy | Both branches exist (`Algorithm5`'s causal + correlational paths) | Worked queries only, no fidelity/accuracy measurement against ground truth | -- | -- | `tests/reasoning/causal/test_algorithm5.py` | `reasoning/causal/algorithm5.py` | Implemented, not validated |
| 17.2 RQ9 | Cognitive cyber reasoning vs. CVSS/ATT&CK | No CVSS/ATT&CK baseline implemented | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ10 | Continual learning vs. static model: planning error, forgetting rate | Both conditions constructible (`mission_adapt` called vs. not) | `benchmarks.convergence` runs the continual-learning condition only; no static-model comparison run; no forgetting-rate measurement | -- | `benchmark_report.json` | `tests/reasoning/continual/test_algorithm7.py` | `reasoning/continual/algorithm7.py` | Implemented, not validated |
| 17.2 RQ11 | TRMG vs. temporal knowledge graph: retrieved-reasoning-chain accuracy | No temporal-knowledge-graph baseline | -- | -- | -- | -- | -- | Unsupported |
| 17.2 RQ12 | MTES/MDRI/CMCS vs. CVSS: correlation with ground-truth consequence | No CVSS baseline / vulnerability database | `scripts/experiment_rq12_cyber_metrics.py` measures MDRI's structural term only | -- | `rq12_cyber_metrics_summary.json` | `tests/cyber/test_metrics.py` | `cyber/metrics.py` | Implemented, not validated (explicitly labeled "partial" by the experiment script itself) |
| 17.2 RQ13 | Full CMI vs. any single-component ablation: superadditivity | Only one of six named ablations implemented (No-EMM) | `evaluation/ablation.py` | -- | `ablation_report.json` | `tests/evaluation/test_ablation.py` | `evaluation/ablation.py` | Implemented, not validated (superadditivity requires all six ablations plus the full-system comparison; only one exists) |
| 17.3 | Six named ablations (No-EMM, no-causal, no-argumentation, no-trust-propagation, no-cyber-propagation, no-continual-learning) + 3 targeted ablations | Only No-EMM implemented | `evaluation/ablation.py` | -- | `ablation_report.json` | `tests/evaluation/test_ablation.py` | `evaluation/ablation.py` | Partially implemented -- 1 of 9 named ablations; the other 8 are Unsupported |
| 17.4 | Testing protocol: unit/integration/adversarial/digital-twin-validation | Unit + integration tests throughout; no adversarial-perturbation test suite | 548 tests pass | -- | -- | Entire `tests/` tree | -- | Partially validated (unit/integration: yes; adversarial: no) |
| 17.4 | Scalability analysis (latency/memory vs. TRMG size, \|M_e\|, \|A\|) | `benchmarks/scalability.py` (2 of 3 named axes: k_hop, \|M_e\|; \|A\| is fixed by the manuscript, not variable) | `benchmarks.scalability` | -- | `benchmark_report.json` | `tests/benchmarks/test_scalability.py` | `benchmarks/scalability.py` | Validated for the two implementable axes; \|A\| axis is Unsupported (nothing varies agent count) |
| 17.4 | Sensitivity, stress, cyber-attack-campaign, mission-interruption, multi-agent-disagreement, graph-evolution, uncertainty-calibration, missing-telemetry, adversarial-perturbation analyses (9 named experiments) | Not implemented as dedicated experiments | -- | -- | -- | -- | -- | Unsupported (9 of 9) |
| 17.5 | Expanded metrics table (16 named metrics) | Trust calibration/ECE implemented; reasoning latency, graph update latency, cyber-impact correlation, mission resilience partially observable from existing outputs; interpretability, human trust, memory efficiency, missing-telemetry/adversarial robustness not implemented | `reasoning/trust/calibration.py`'s ECE is real and tested | -- | -- | `tests/trust/test_calibration.py` | `reasoning/trust/calibration.py` | Mixed -- 1 of 16 (trust calibration/ECE) Validated as a reusable metric; the rest Unsupported as executed measurements |
| 17.6 | Statistical analysis plan: >=30 seeds, 5-fold CV, paired t-test/Wilcoxon + Holm-Bonferroni, effect sizes | Machinery implemented (`evaluation/statistics.py`); protocol not executed for any RQ | -- | -- | -- | `tests/evaluation/test_statistics.py` | `evaluation/statistics.py` | Implemented, not validated -- the machinery is real and tested; no RQ has been run under this specific protocol (>=30 seeds, 5-fold CV) |
| 17.7 | "No scenario, baseline, or ablation has yet been executed... this is the immediate next phase" | N/A -- manuscript's own honesty statement | Phases 7-9 executed many scenarios/one ablation since this was written | -- | -- | -- | manuscript Section 17.7 | Partially superseded, correctly documented -- this repository now *has* executed real scenarios/experiments/one ablation (unlike when Section 17.7 was written), but still has not executed any full RQ-with-baseline comparison; see `docs/manuscript-consistency-report.md` for the precise wording gap this creates |

## Section 18-19 -- Limitations and Discussion

| # | Claim | Implementation | Experiment(s) | Figure(s) | Table(s) | Test(s) | Source | Status |
|---|---|---|---|---|---|---|---|---|
| 18 | Six named limitations (no Section 17 evaluation yet; TRMG/EMM growth needs calibration; credibility weights could over-trust; causal discovery not solved; continual-learning regularization unvalidated; cyber metrics need calibration) | All six remain accurate today | See per-item rows above | -- | -- | -- | manuscript Section 18 | Validated as still-accurate self-assessment -- Phase 9's own audit (this document) confirms each one still holds, with the nuance that *some* real, partial evidence now exists where none did when Section 18 was written |
| 19.4 | Generalization to ICS, spacecraft, autonomous vehicles, critical infrastructure, medical CPS, power grids, smart factories | Not implemented; aircraft is the only domain instantiated anywhere in code | -- | -- | -- | -- | -- | Future work (manuscript's own framing; no code claims otherwise) |

## Summary counts

| Status | Approximate row count |
|---|---|
| Validated | 33 |
| Implemented, not validated | 14 |
| Future work (manuscript's own classification) | 3 |
| Unsupported | 15 |

Counts are per-row above (a "row" is one claim, not one manuscript
sentence); several rows bundle closely related sub-claims (e.g. Section
17.4's nine unimplemented experiments are one row). See
`docs/negative-results.md` for the specific negative findings among the
"Validated" rows (a claim can be *validated as executed* while the result
itself is unfavorable -- RQ6's ablation is the clearest example).
