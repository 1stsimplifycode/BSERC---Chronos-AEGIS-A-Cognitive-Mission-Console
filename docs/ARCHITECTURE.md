# Architecture: Paper-to-Repository Map

Source of truth: `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf`.

This document maps every major component the manuscript defines to its
implementation location and current status. "Implemented" means real,
tested, faithful code exists (see `tests/`). "Interface only" means a
dependency-injection contract and an honest placeholder exist, but the
component's actual logic does not. "Not started" means neither exists yet.

| Manuscript component | Section | Location | Status |
|---|---|---|---|
| CMI hierarchy / Cognitive Mission State sigma(m,t) | 4.1 | `reasoning/state.py` | Implemented |
| Mission Utility U(sigma) | 4.2 | `reasoning/utility.py` | Implemented |
| TTRN pipeline / Algorithm 1 | 5 | `reasoning/ttrn/pipeline.py` | Implemented -- every dependency-injected stage now has a real implementation |
| Temporal Trust Graph (TTG) | 6.1 | `graph/trmg/model.py` (`EdgeAttributes`) | Implemented |
| Temporal Reasoning Memory Graph (TRMG) | 6.2 | `graph/trmg/model.py` (`NodeHistory`, `TemporalReasoningMemoryGraph`) | Implemented |
| Temporal Dependency Analysis | 6.5 | `graph/trmg/temporal.py` | Implemented |
| Mission-Aware GNN | 7.1 | `reasoning/trust/gnn.py` | Implemented (architecture only -- untrained; see module docstring) |
| Trust Propagation | 7.2 | `reasoning/trust/propagation.py` | Implemented |
| Bayesian Trust Propagation (posterior update) | 7.3 | `reasoning/trust/bayesian.py` | Implemented |
| Mission Criticality Model | 7.4 | `reasoning/trust/criticality.py` | Implemented |
| Episodic Mission Memory (EMM) | 8 | `memory/` | Implemented (see `memory/README.md` for documented simplifications) |
| Multi-Agent Hypothesis Generation | 9.1 | `agents/hypothesis.py`, `agents/roster.py` | Data structures + agent roster implemented; concrete domain reasoning per agent not started |
| Structured AI Argumentation | 9.2-9.7 | `agents/argumentation/` | Implemented (see `agents/README.md` for documented simplifications) |
| Causal Mission Intelligence | 10 | `reasoning/causal/` | Implemented (see `reasoning/causal/README.md` for documented scope limitations) |
| Cognitive Cyber Reasoning | 11 | `cyber/cognitive_reasoning.py`, `cyber/consequence.py` | Implemented (see `cyber/README.md` for documented assumptions) |
| Mission-Aware Cyber Risk Propagation, attack chains, supply-chain/zero-trust/adversarial reasoning | 12 | `cyber/risk_propagation.py`, `cyber/attack_chain.py`, `cyber/robustness.py` | Implemented (see `cyber/README.md`; two of Section 12.5's four robustness mitigations are satisfied by reusing Sections 6/9 rather than new code) |
| MTES / MDRI / CMCS | 12.6 | `cyber/metrics.py`, `cyber/centrality.py` | Implemented (see `cyber/README.md` for documented assumptions; requires empirical calibration per the manuscript's own text, not claimed here) |
| Temporal Attack Graphs | 13 | `cyber/attack_graph.py` | Implemented, integrated into the TRMG rather than a separate graph (see `cyber/README.md`) |
| AI Patch Scheduler | 14 | `scheduler/` | Implemented; both of Algorithm 6's cost terms are now real -- MissionTrustLoss via the Section 10.5 causal engine, CyberExposure via Section 12.6.1's MTES (`cyber/scheduler_integration.py`) -- no placeholder values remain |
| Mission Digital Twin | 15 | `digital_twin/` | Implemented -- a real discrete-event simulation engine (clock, event queue, 17 named event types across the original 12 plus Phase 8's firmware scenarios, 18 scenarios, deterministic replay), producing the same TRMG the rest of the framework reads from (see `digital_twin/README.md`) |
| Continual Learning loop | 16 | `reasoning/continual/` | Implemented -- Algorithm 7 (MissionAdapt) and the Section 16.2 Regularized Continual Objective, both against real, analytically-verified gradients (see `docs/continual-learning-guide.md` for scope decisions and two honest findings discovered while implementing it) |
| Evaluation Protocol | 17 | `evaluation/`, `scripts/`, `benchmarks/` | Infrastructure implemented (batch sweeps, aggregation, statistics, visualization, one real ablation, a benchmark suite, a one-command reproducibility workflow); no RQ's full evaluation has been run -- every named baseline (RQ2/RQ5/RQ8/RQ9/RQ12) is still missing. Real, non-fabricated experiments have been run for RQ6/RQ7/RQ10/RQ12 (partial) -- see `evaluation/README.md`, `docs/benchmark-guide.md` |

## Every Algorithm 1 stage is now real

`reasoning/ttrn/pipeline.py`'s `TTRNPipeline` implements every line of
Algorithm 1's control flow. `EpisodicMemoryInterface`,
`ArgumentationReconcilerInterface`, `CausalEngineInterface`, and
`PatchSchedulerInterface` all now have real implementations
(`memory.emm.EpisodicMissionMemory`,
`agents.argumentation.aggregation.StructuredArgumentAggregator`,
`reasoning.causal.algorithm5.StructuralCausalEngine`,
`scheduler.algorithm6.AIPatchScheduler`) in addition to their `Null*`
opt-outs, which either return the honest empty result for the trivial case
(e.g. zero hypotheses -> no decision to reconcile; zero candidate
interventions -> no causal context; no `patch_scheduling_request` -> lines
12-13 silently skipped) or raise `NotImplementedError`, which the pipeline
catches and records on `TTRNResult.warnings` rather than fabricating a
value or crashing when a `Null*` is used instead. See
`examples/run_ttrn_pipeline.py` (no optional backends),
`examples/run_ttrn_pipeline_with_memory.py` (real EMM),
`examples/run_ttrn_pipeline_with_argumentation.py` (real argumentation,
including an escalation case), `examples/run_ttrn_pipeline_with_causal.py`
(real causal engine + the eight-level Explainability Pipeline), and
`examples/run_ttrn_pipeline_with_scheduler.py` (the full capstone -- real
memory, argumentation, causal engine, *and* patch scheduler together,
reproducing the manuscript's own worked Section 14.1 scenario) for what
this looks like end to end, including the printed warnings.

This was a deliberate design choice, not a workaround: it let the graph,
trust-propagation, GNN/Bayesian, memory, argumentation, causal, and
scheduling layers be built, tested, and used for real incrementally.
Sections 8, 9, 10, and 14's real implementations all went through the same
path -- interface-only placeholder to real implementation -- without any
pipeline call site needing to change its shape.

Sections 11-13 (`cyber/`, Phase 6) are not a fifth `TTRNPipeline`
dependency-injection point -- Algorithm 1 does not name a "cyber reasoning"
stage the way it names memory, argumentation, the causal engine, or the
scheduler -- but they are not a separate subsystem either: `cyber/` reuses
Section 8's `EpisodicMemoryStore` (Algorithm 3, extended with a predicate
filter), Section 7.2's `rho(u->v,t)` (extended to multiple hops), Section
9's argumentation (for sensor-spoofing conflict, unchanged), and now feeds
Algorithm 6's CyberExposure term directly (`cyber/scheduler_integration.py`,
replacing Phase 5's caller-supplied placeholder) and the Explainability
Pipeline's Evidence- and Graph-levels (`reasoning/causal/explainability.py`).
See `examples/run_ttrn_pipeline_with_scheduler.py` for cyber reasoning
composed into a full Algorithm 1 run end to end.

## The Digital Twin is now the producer, Algorithm 1 the consumer

`digital_twin/` (Section 15, Phase 7) is a real discrete-event simulation
engine -- not a visualization, not a game -- that owns one real TRMG and
advances it by applying scheduled, immutable `SimulationEvent`s in time
order. It duplicates no state: everything Section 15 lists as simulated
(aircraft state, telemetry, comms, software, firmware, sensors, weather,
cyber incidents, trust evolution) already lives as TRMG nodes/edges/
history (Section 6); only mission phase, objective/operator/cyber-incident
logs, and terminal status -- which have no TRMG-node analogue -- live in
the twin's own `MissionEnvironment`.

`digital_twin/integration.py`'s `run_scenario_with_pipeline` is the one
seam between the two: it schedules a `ScenarioSpec`'s events onto a twin
and then calls `TTRNPipeline.infer` (Algorithm 1, Section 5) against that
same twin's graph, unchanged -- exactly as Algorithm 1 would be called
against a hand-built graph in `examples/`. From there, the full
TRMG -> Memory -> Argumentation -> Causal Engine -> Scheduler ->
Explainability chain is Algorithm 1's own control flow, untouched by this
package. `digital_twin/dataset.py` flattens the resulting `TTRNResult`s
into a reproducibility dataset (CSV/JSON); `evaluation/` sweeps this over
scenario x seed x parameter grids and aggregates the results; two runnable
scripts under `scripts/` demonstrate this end to end against real (not
hand-picked) RQ-adjacent questions -- see `evaluation/README.md`.

What remains (every named RQ baseline Section 17 needs) has no
implementation.

## Section 16 closes the loop: Algorithm 7 runs after Algorithm 1

`reasoning/continual/` (Phase 8) implements Algorithm 7 (MissionAdapt,
Section 16.5) as a separate call a caller makes once a mission concludes,
*after* `TTRNPipeline.infer(..., outcome=..., embedding=...)` has already
run (Algorithm 1 line 15's EMM-Update already satisfies Algorithm 7 line
1 -- the same write, not a duplicate one). `mission_adapt` then performs
lines 2-8: TRMG decay, an experience-replay batch drawn from Episodic
Memory, the Section 16.2 regularized gradient step against
`MissionAwareGNN`'s six prediction heads, Fisher-information recompute,
and agent-credibility/Meta-Memory update (reusing `MetaMemory.recompute`
directly). See `docs/continual-learning-guide.md` for the two honest
findings this implementation surfaced (the EWC penalty's gradient is
structurally zero under single-step-per-mission Algorithm 7; retrieved
memory has no path into the GNN's predictions at all) and
`benchmarks.convergence` for measuring it over a real mission sequence.

## Phase 8 also found and fixed a real cross-process determinism bug

`digital_twin/twin.py`'s `node_feature_vector` derived its per-node RNG
seed via Python's built-in `hash((seed, node_id))`, which is salted per
*process* by `PYTHONHASHSEED` by default -- the same seed and node id
produced a different feature vector in every separate process
invocation, while appearing perfectly deterministic within any single
process (which is why no test in Phases 1-7 caught it: pytest runs an
entire session in one process). This was only found while verifying
Phase 8's reproducibility-workflow claim by hashing
`scripts/reproduce_all.py`'s output across two separate `python`
invocations with the same seeds. Fixed by deriving the seed via
`hashlib.sha256` instead. See `docs/reproducibility-guide.md` for the
full account and the subprocess-based regression test that guards it.

## Notation note

The source PDF's text extraction strips some Greek glyphs. Where the
manuscript's prose makes the intended symbol and its role unambiguous (e.g.
"T_post(m,t) and [uncertainty] are the Bayesian trust posterior and its
uncertainty"), the code uses a descriptive Python identifier
(`uncertainty`, `trust_posterior`, `kappa_m`, `alpha_base`, ...) instead of
guessing at a specific stripped glyph. Every such mapping is documented in
the docstring of the module that introduces it (see `reasoning/state.py`
for the most detailed example).
