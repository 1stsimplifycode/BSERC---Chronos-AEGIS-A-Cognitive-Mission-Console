# Negative Results (Phase 9)

This document consolidates every genuine negative or unfavorable finding
discovered while implementing and benchmarking Chronos-AEGIS. None of
these were removed, softened, or hidden elsewhere in the repository or
in the manuscript's own text -- each is preserved here with why it
occurs, whether it represents a limitation, and whether it motivates
future work. "Negative" here means: a measurement that did not turn out
favorably, or a mechanism that does not do what a first reading of the
manuscript might suggest -- not a bug (each item below is the *correct*
behavior of the code as written; where a bug *was* found, it is listed
separately in "Bugs found and fixed," not here, because those were
corrected rather than left as standing findings).

## 1. The EWC penalty's gradient is structurally zero (Section 16.2)

**What was found**: In every `benchmarks.convergence` run, `ewc_penalty`
is exactly `0.0` at every mission, for every scenario sequence tested.

**Why it occurs**: `reasoning/continual/algorithm7.py`'s `mission_adapt`
takes exactly one gradient step per call (Algorithm 7 line 4 is a single
step, not an inner training loop), and `state.reference_heads` (theta\*)
is set, every call, to the theta that same call just produced (line 8's
"return updated theta" becomes the next call's theta\*). So at the start
of every call after the first, `heads == reference_heads` exactly -- and
both the penalty's value and its gradient
(`2*lambda*F*(heads-reference)`) are identically zero at that evaluation
point. This is a real mathematical consequence of the single-step-per-
mission design that Algorithm 7's literal eight lines specify, not a
coding defect: verified analytically (the gradient computation itself is
checked against finite differences to ~1e-11,
`tests/reasoning/continual/test_objective.py`) and empirically (every
convergence benchmark run).

**Is it a limitation?** Yes. Under this literal reading, the
regularization term Section 16.2 introduces specifically to mitigate
catastrophic forgetting currently has no effect on any parameter update.

**Does it motivate future work?** Yes, and the manuscript already
anticipates needing this work: Section 16.6 states "we have not evaluated
whether the specific regularization strength lambda and replay-batch
composition prevent catastrophic forgetting in practice for this
architecture." Resolving this negative finding would require a design
decision the manuscript does not make explicitly -- e.g., anchoring
theta\* to a periodic checkpoint (every N missions) rather than every
single call, or taking multiple gradient steps per mission before
advancing theta\*. Either is a legitimate reading of an
under-specified algorithm; picking one was out of scope for the phase
that discovered this (implementing Algorithm 7 exactly as written,
not amending it) and remains open.

See `reasoning/continual/algorithm7.py`'s own docstring for the
line-by-line account, and `docs/continual-learning-guide.md`.

## 2. Retrieved episodic memory never reaches the GNN's predictions

**What was found**: `evaluation/ablation.py`'s No-EMM ablation (a real,
executed comparison of an `EpisodicMissionMemory`-enabled `TTRNPipeline`
against the bare default, same scenarios/seeds) shows **zero** difference
in every Section 7.1 prediction-head metric (`trust_mean`, `p_success`,
`expected_delay`, `cyber_impact`, `comm_reliability`,
`operational_confidence`) -- Cohen's d = 0.0 exactly for all six.

**Why it occurs**: `reasoning/ttrn/pipeline.py`'s `TTRNPipeline.infer`
calls `MissionAwareGNN.embed` without ever passing
`retrieved_memory_features`. That parameter exists on `embed` --
Section 7.1's own "N_k(m) union M_ret" attention pool names retrieved
memory as part of what the GNN should attend over -- but nothing in the
pipeline populates it from the `MemoryRetrievalResult` Algorithm 1 line 3
(EMM-Retrieve) already returns. Only memory-specific bookkeeping
(`memory_episodes_retrieved`, `warning_count`) differs between the two
conditions (Cohen's d = 6.51 for `memory_episodes_retrieved` -- a real,
large, but structurally uninteresting difference).

**Is it a limitation?** Yes, a real architectural gap between the
formalism (Section 7.1) and the implementation.

**Does it motivate future work?** Yes: bridging `MemoryRetrievalResult`
into `embed`'s attention pool needs a feature encoder that converts a
`RankedEpisodicCandidate`'s categorical/continuous `FeatureVector` (or,
after Phase 8, its stored `embedding`) into the same embedding space the
GNN's attention mechanism already operates in. This is a genuine new
design decision, not a bug fix, and was out of scope for every phase to
date. `memory/README.md`'s "What's still not connected" section has
flagged this since Phase 2; Phase 8's ablation independently
re-confirmed it with a real measurement rather than leaving it as an
assertion.

## 3. Episodic Memory retrieval originally scored below a majority-class baseline (RQ6/RQ7) -- and its fix

**What was found (Phase 7, first measurement)**: precision@1 = 0.3846
against a majority-class baseline of 0.6154, over the original
13-scenario library, 130 episodes -- default, unweighted similarity
retrieval performed *worse* than always guessing the most common
outcome.

**Why it occurred**: Diagnosed, not guessed: of Section 8.2.3's six
categorical fields, the Digital Twin's event catalog only ever set two
of them (`weather_category`, `entry_node_type`); `continuous_features`
was entirely unpopulated by `extract_feature_vector`. Nine of the
thirteen scenario types therefore produced a completely generic feature
vector, indistinguishable from each other regardless of their actual
(different) outcomes, so similarity-based ranking among them was
effectively arbitrary.

**The fix (Phase 8, manuscript-consistent, not fabricated)**:
`memory/features.py`'s `extract_feature_vector` now also reads
incoming-edge confidence (chi(e).c, Section 6.1 -- an unambiguous
"salient TRMG attribute," Section 8.2.3's own phrase) into
`continuous_features`.

**Measured result (Phase 8, current 18-scenario library, controlled
before/after on identical stored episodes)**: BEFORE (categorical fields
only) = 0.5556, AFTER (+ incoming-edge confidence) = 0.7778, against a
0.5000 majority-class baseline -- now favorable. Re-run
`scripts/experiment_rq6_rq7_memory_retrieval.py` to verify.

**Is it (still) a limitation?** Partially. The fix closes the specific
feature-coverage gap that caused the original regression, but Section
8.6's own framing -- that the similarity weights themselves are an open
empirical question (RQ6, RQ7) -- remains true: `SimilarityWeights` is
still the unweighted default, not a fitted/learned set of weights.

**Does it motivate future work?** Yes: calibrating the weights
themselves (as opposed to fixing feature coverage) is exactly what
RQ6/RQ7 ask for and remains undone.

## 4. MDRI's own value is 0.0 for every scenario, canonical and firmware alike

**What was found**: `scripts/experiment_rq12_cyber_metrics.py` --
`mdri_value` is exactly `0.0` for all fifteen scenarios run (10 canonical
+ 5 firmware), across every seed.

**Why it occurs**: MDRI(m) = Sum_v rho(v)\*centrality(v) (Section
12.6.2). `rho(v)` (a dependency node's own intrinsic vulnerability risk)
is supplied via `intrinsic_risk`, which `digital_twin/integration.py`
always passes as `{}` -- this repository has no vulnerability database
(no SBOM/CVE ingestion exists anywhere), so every dependency node's own
risk is honestly 0 rather than fabricated. With every rho(v) = 0, the sum
is 0 regardless of centrality.

**The related, now-fixed sub-finding (Phase 7 -\> Phase 8)**: `|Dep(m)|`
(the *structural* term, independent of `intrinsic_risk`) was also
provably 0 for every canonical scenario, because none of them schedule a
`FirmwareUpdate` event (the only event type that sets the
`dependency_identity` attribute `Dep(m)` is keyed on). Phase 8's five new
firmware scenarios close this specific gap: `|Dep(m)| = 1.0` for all five,
confirmed by a hand-built graph separately proving MDRI's formula is
genuinely non-zero (0.6) once both a populated `Dep(m)` *and* non-zero
`intrinsic_risk` are present (`scripts/experiment_rq12_cyber_metrics.py`'s
`_demonstrate_nonzero_mdri`).

**Is it a limitation?** Yes -- `mdri_value` itself remains uninformative
on any Digital Twin scenario today.

**Does it motivate future work?** Yes, and the manuscript already
anticipates it: Section 18 states the three new cyber metrics (MTES,
MDRI, CMCS) are "formal definitions requiring empirical calibration
against ground-truth mission consequence before they can be trusted to
outperform CVSS in practice." A vulnerability database (or a documented,
non-fabricated proxy for one) is the concrete prerequisite; none exists
in this repository, and none is fabricated as a placeholder.

## 5. Optional-dependency-gated code paths are unverified in the audited environment

**What was found**: `digital_twin/visualization.py` (19% line coverage)
and the scipy success-path in `evaluation.statistics.paired_comparison`
(lines 154-163 uncovered) are not exercised in the environment this audit
ran in, because matplotlib and scipy are not installed.

**Why it occurs**: Both are genuinely optional dependencies (documented
in each module's own docstring, and in `README.md`'s "Optional
dependencies" section); no test suite run depends on them being present,
by design -- the failure-path behavior (a clear `ImportError`, or a
`None` p-value with an explanatory note) *is* tested and covered at
100%, matching the project's established honesty convention for optional
components.

**Is it a limitation?** Only in the narrow sense that the success paths
(a real rendered figure; a real scipy-computed p-value) have not been
exercised by an automated test in this specific audit environment. They
are not unimplemented -- installing either dependency and re-running the
test suite exercises them (see `docs/reproducibility-audit.md`).

**Does it motivate future work?** Only environment-level: install scipy/
matplotlib in CI (or a dedicated "optional dependencies" CI job) to get
automated coverage of both success paths. No code change is implied.

## 6. Bugs found and fixed (for context, not standing findings)

These are not negative *results* -- they were defects, and are already
corrected with regression tests. Listed here only so a reader auditing
"what went wrong during development" has the complete picture in one
place, cross-referenced to where the fix lives.

- **Cross-process determinism bug** (`digital_twin/twin.py`,
  `node_feature_vector`): used Python's built-in `hash()`, salted per
  process by `PYTHONHASHSEED` by default -- looked perfectly
  deterministic within any single process (including a full pytest run)
  while silently differing across separate `python` invocations with the
  same seed. Found while verifying `scripts/reproduce_all.py`'s
  reproducibility claim (Phase 8); fixed via `hashlib.sha256`. Regression
  test: `tests/scripts/test_reproduce_all.py::test_node_feature_vector_is_deterministic_across_separate_processes`
  (spawns two real subprocesses). See `docs/reproducibility-guide.md`.
- **Tuple-vs-list stripping bug** (`scripts/reproduce_all.py`,
  `_strip_nondeterministic`): `dataclasses.asdict()` preserves
  tuple-typed fields as tuples, not lists; the stripper originally only
  handled `dict`/`list`, so timing fields nested inside a tuple silently
  escaped stripping before hashing. Fixed by handling `(list, tuple)`
  together. Regression test:
  `tests/scripts/test_reproduce_all.py::test_strip_nondeterministic_recurses_into_tuples`.

## Summary

Five standing negative findings (Sections 1-5 above) remain in this
repository by design -- none was removed from the manuscript's framing or
papered over in code. Three are direct, real measurements
(`evaluation/ablation.py`, `scripts/experiment_rq6_rq7_memory_retrieval.py`,
`scripts/experiment_rq12_cyber_metrics.py`); two are structural/
environmental (the EWC-gradient finding; the optional-dependency
coverage gap). Every one of them either corresponds to an open question
the manuscript itself already names (Sections 16.6, 8.6, 18) or is
newly discovered and documented at the point it was found, with a
concrete, undone next step rather than a vague caveat.
