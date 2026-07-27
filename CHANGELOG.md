# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/) once a
first tagged release exists (`v1.0.0`, not yet cut -- see
`RELEASE_NOTES.md`). Entries are grouped by development phase, the
project's own unit of work, since no version has been tagged yet; the
first six phases correspond to git commits `9f76152`..`e1e08ea`
(`git log --oneline`), and Phases 7-9 are the current, uncommitted
working-tree state as of this entry.

## [Unreleased]

### Backend & Deployment Preparation (post-Phase-9)
- Added `backend/app.py`: a minimal, real FastAPI service wrapping
  `TTRNPipeline` and the Digital Twin scenario library directly (health
  check, scenario listing, live scenario-run endpoint returning the full
  `MissionRecord` trajectory) -- not the full production layer
  (auth/datastores/workers) `backend/README.md` originally envisioned in
  Phase 1, which remains future work.
- Added `deployment/render.yaml`; the exact configured start command was
  verified locally against a real running server before being written
  here.
- Added `tests/backend/test_app.py` (27 tests, incl. a parametrized run
  of all 18 scenarios through the live API).
- Added `fastapi`/`uvicorn` as a new `backend` optional-dependency group;
  put the previously-unused `pydantic` dependency to real use (API
  response schemas) for the first time (see `docs/code-quality-audit.md`
  finding 2, now partially resolved).
- Repository pushed to GitHub: https://github.com/1stsimplifycode/chronos-aegis.
  No live Render deployment or URL yet -- creating one requires the
  repository owner's own Render account; see `deployment/README.md`.
- Added `frontend/`: a minimal, build-step-free static console
  (`index.html`/`styles.css`/`app.js`, no npm/bundler/CDN dependency)
  mounted at `/ui` by `backend/app.py`. Lists all 18 scenarios, runs one
  live against the real API, and renders the resulting trajectory as a
  hand-drawn canvas chart and table -- not the React/Next.js "Enterprise
  Dashboard" `frontend/README.md` originally envisioned, which remains
  future work. Added 2 tests confirming the static mount actually serves
  through the live app (`tests/backend/test_app.py`, now 29 tests total).

### Phase 9 -- Research Validation, Artifact Audit & Journal Packaging
- Added `docs/traceability-matrix.md`: a per-claim manuscript-to-evidence
  mapping with honest classification (Validated / Implemented-not-
  validated / Future work / Unsupported).
- Added `docs/manuscript-consistency-report.md`, `docs/negative-results.md`,
  `docs/reproducibility-audit.md`, `docs/code-quality-audit.md`,
  `docs/documentation-review.md`.
- Added `paper/artifact/`: the journal/artifact-evaluation package
  (artifact description, reviewer quick-start, experiment guide,
  benchmark appendix, reproducibility appendix, implementation appendix,
  supplementary material).
- Added `graph/README.md`, `reasoning/README.md`, `tests/README.md`
  (previously missing top-level package documentation); added usage
  examples to `agents/README.md` and `cyber/README.md`; corrected one
  stale explanatory claim in `cyber/README.md`.
- Fixed an outdated Phase-1-era comment/docstring in
  `examples/run_ttrn_pipeline.py` that misleadingly implied later-phase
  components (memory, argumentation, causal, scheduler) were not
  implemented in the repository, rather than simply not configured for
  that specific minimal example.
- Added `requirements-lock.txt` (pinned dependency versions for exact
  environment reproduction).
- Documented (not removed) two declared-but-unused dependencies
  (`pydantic`, `PyYAML`) and one unloaded configuration file
  (`configs/default.yaml`), each reserved for already-planned future
  components.
- Added `LICENSE` (MIT), `CITATION.cff`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`.
- No new algorithms, modules, or architectural changes -- this phase was
  audit and packaging only, per its own brief.

### Phase 8 -- Continual Learning, Benchmarking & Publication Readiness
- Implemented Section 16 (Adaptive Learning and Continual Reasoning):
  `reasoning/continual/` -- the Regularized Continual Objective (Section
  16.2, with an analytic gradient verified against finite differences)
  and Algorithm 7 (MissionAdapt, Section 16.5).
- Added Episodic Learning: `memory/reliability.py`'s non-destructive,
  outcome-driven retrieval-confidence adjustment; `EpisodicEntry` gained
  an `embedding` field for experience replay (Section 16.3).
- Added TRMG Evolution: `graph/trmg/snapshot.py` and
  `TemporalReasoningMemoryGraph.snapshot()`/`.restore()`/`.prune_history()`.
- Added evidence-driven Trust Calibration (`reasoning/trust/calibration.py`,
  ECE + histogram-binning recalibration) and Scheduler Learning
  (`scheduler/calibration.py`, evidence-driven `w_cyber`/`w_trust`
  calibration preserving Algorithm 6's argmin unmodified).
- Diagnosed and fixed a real Episodic Memory retrieval-precision
  regression (RQ6/RQ7): `memory/features.py` now reads incoming-edge
  confidence into `continuous_features`; precision@1 improved from below
  a majority-class baseline to above it (measured, not assumed).
- Added five new Digital Twin scenarios
  (`digital_twin.scenarios.FIRMWARE_SCENARIO_BUILDERS`) exercising
  MDRI's dependency term for the first time; the original ten canonical
  scenarios were left unmodified.
- Added `benchmarks/` (scalability, continual-learning convergence,
  retrieval accuracy, scheduler-quality benchmarks) and
  `scripts/reproduce_all.py` (one-command reproducibility workflow).
- Found and fixed a real cross-process determinism bug in
  `digital_twin/twin.py` (`hash()` salted by `PYTHONHASHSEED`; fixed via
  `hashlib.sha256`), with a subprocess-based regression test.

### Phase 7 -- Digital Twin & Experimental Infrastructure
- Implemented Section 15 (Mission Digital Twin): `digital_twin/` --
  discrete-event simulation engine, 12 named event types, Section
  15.1's ten canonical scenarios, deterministic replay.
- Added `evaluation/` (batch sweeps, aggregation, statistics) and two
  experiment scripts (`scripts/experiment_rq12_cyber_metrics.py`,
  `scripts/experiment_rq6_rq7_memory_retrieval.py`).
- Added `digital_twin/visualization.py` (matplotlib-optional figures).

### Phase 6 -- Sections 11-13 (git `e1e08ea`)
- Implemented Cognitive Cyber Reasoning, Mission-Aware Cyber Risk
  Propagation, and the Temporal Attack Graph; closed every remaining
  placeholder in the AI Patch Scheduler's cost terms.

### Phase 5 -- Section 14 (git `9f451df`)
- Implemented the AI Patch Scheduler, Algorithm 6.

### Phase 4 -- Section 10 (git `39977b7`)
- Implemented Causal Mission Intelligence: the Structural Causal Model,
  Algorithm 5, and the Explainability Pipeline.

### Phase 3 -- Section 9 (git `38c3397`)
- Implemented Structured AI Argumentation: the Argumentation Graph,
  Dung-style semantics, and Algorithm 4.

### Phase 2 -- Section 8 (git `3764c5d`)
- Implemented Episodic Mission Memory: the four memory tiers, Algorithm
  2, and Algorithm 3.

### Phase 1 -- Repository skeleton, Sections 4-7 (git `9f76152`)
- Established the manuscript-as-source-of-truth convention.
- Implemented the Cognitive Mission State, the TTRN pipeline (Algorithm
  1), the Temporal Reasoning Memory Graph, the Mission-Aware GNN, Trust
  Propagation, the Bayesian posterior update, and the Mission
  Criticality Model.
