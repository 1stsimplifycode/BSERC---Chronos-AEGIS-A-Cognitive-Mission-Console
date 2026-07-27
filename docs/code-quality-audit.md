# Code-Quality Audit (Phase 9)

Scope: dead code, duplicated implementations, undocumented assumptions,
inconsistent naming, outdated comments, unused configuration, stale
scripts, obsolete experiments. Per this phase's brief, only demonstrably
obsolete code was removed; everything else found is documented here.

## Findings

### 1. Outdated comment (fixed)

`examples/run_ttrn_pipeline.py`'s module docstring and a runtime `print`
statement said "everything implemented in Phase 1" / "Stages NOT
implemented in this phase," dating from when Episodic Memory,
Argumentation, the Causal Engine, and the Scheduler genuinely did not
exist yet. All four have existed since Phases 2-6; this example simply
does not wire them in (by design, to isolate Algorithm 1's core stages).
The wording was misleading to a reader running this exact example and
reading the literal output. **Fixed**: reworded to "no optional
collaborators configured" / "Optional stages not configured for this
example," with cross-references to the four incremental examples and
`docs/ARCHITECTURE.md`. Verified: the example still runs and prints the
corrected text; `ruff check examples` remains clean.

### 2. Two declared dependencies are never imported (not removed -- reserved for planned work)

`pyproject.toml`'s `[project.dependencies]` lists `pydantic>=2.6` and
`PyYAML>=6.0` alongside `numpy>=1.26`. A repository-wide grep
(`grep -rl "^import pydantic\|^from pydantic"` / same for `yaml`) found
**zero** import sites for either package anywhere in `graph/`,
`reasoning/`, `memory/`, `agents/`, `cyber/`, `scheduler/`,
`digital_twin/`, `evaluation/`, `benchmarks/`, or `scripts/` (`numpy`, by
contrast, is imported in 23 files). Both are genuinely unused today.

**Not removed**, because both are reserved for explicitly-planned,
not-yet-built components named elsewhere in this repository's own
documentation, not leftover cruft:
- `PyYAML` pairs with `configs/default.yaml` (see finding 3 below) --
  the natural library for loading it once a config-loading mechanism is
  built.
- `pydantic` is the conventional schema library for a FastAPI backend
  (`backend/README.md`: "not yet implemented... Per the project's
  deployment requirements").

Removing either now would be presumptuous given both are pre-declared
for work already on this repository's own roadmap; a reviewer should
simply know they are currently inert, which this entry documents.

### 3. `configs/default.yaml` is not loaded by any code (unused configuration)

`configs/default.yaml` documents default parameter values with direct
manuscript citations (`k_hop_bound: 6` -\> Section 5.1; `default_decay_rate:
0.05` -\> Section 6.1; `alpha_base: 0.5` -\> Section 7.4; the five Section
4.2 utility weights; `default_prior_variance: 1.0`). No Python module
reads this file -- every one of these defaults is instead hardcoded
directly in the relevant function/class signature (e.g.
`TTRNPipeline.__init__(..., k_hop: int = 6, ...)`,
`EdgeAttributes(..., decay_rate: float = 0.05, ...)`).

Spot-checked for drift: every value in `configs/default.yaml` matches
its corresponding hardcoded default in code as of this audit (`k_hop=6`
in `reasoning/ttrn/pipeline.py`; `decay_rate=0.05` in
`graph/trmg/model.py`'s `EdgeAttributes`; `alpha_base=0.5` in
`reasoning/trust/criticality.py`). The file is accurate, not stale --
just disconnected from a config-loading mechanism that does not exist
yet. Not removed, for the same "reserved for planned work" reason as
finding 2; documented here so the disconnect is explicit rather than
discovered by surprise.

### 4. `reasoning/utility.py` (Mission Utility U(sigma), Section 4.2) is implemented, tested, and never called outside its own test

A repository-wide import search found `reasoning.utility` imported only
by `tests/reasoning/test_utility.py` -- no pipeline stage, example, or
experiment computes U(sigma) at runtime. This is consistent with the
manuscript's own framing (Section 4.2's own text: U(sigma) is "what every
CMI mechanism is framed against theoretically," a unifying conceptual
objective, not a literal per-query computation Algorithm 1 names as one
of its sixteen lines) -- already reflected in
`docs/traceability-matrix.md`'s Section 4.2 row ("not used to drive an
actual optimization/decision anywhere"). Not dead code (it is correct
and tested against the manuscript's formula); flagged here as "real but
unwired," and left as-is -- wiring it into a live decision point would be
a new feature, out of scope for this phase.

### 5. Minor, documented duplication: sigmoid/softplus squashing

`reasoning/trust/gnn.py`'s `MissionAwareGNN.predict` defines local
`sigmoid`/`softplus` closures; `reasoning/continual/objective.py`'s
`_head_prediction_and_local_gradient` re-implements the same two
functions (needed to compute their analytic derivatives for Algorithm
7's gradient). This is a real, small duplication, but it is already
explicitly documented at the point of duplication
(`_head_prediction_and_local_gradient`'s own docstring: "Mirrors
`MissionAwareGNN.predict`'s squashing choices exactly") and verified
correct via a finite-difference check
(`tests/reasoning/continual/test_objective.py`). Consolidating it into a
shared helper would create a new cross-module coupling between the GNN
and the continual-learning objective for two one-line functions -- judged
not worth the coupling; left as documented duplication rather than
"fixed."

### 6. No TODO/FIXME/XXX/HACK markers anywhere in the source tree

`grep -rn "TODO\|FIXME\|XXX\|HACK"` across every `.py` file (excluding
`.venv`) returned zero matches. No deferred-work markers were left
behind by any phase.

### 7. Stub directories (`backend/`, `frontend/`, `deployment/`, `docker/`) remain accurate

Each still states "Status: not yet implemented" and describes only
intended future content -- verified still true (no code exists in any of
the four). Not stale.

### 8. No stale or obsolete example/experiment scripts found

All five `examples/run_ttrn_pipeline*.py` scripts were spot-run during
this audit (`run_ttrn_pipeline.py` directly; the other four via the
existing test suite's own execution of equivalent pipeline
configurations) and produce output consistent with their docstrings.
Both `scripts/experiment_*.py` and `scripts/reproduce_all.py` were
executed fresh during this audit (`docs/reproducibility-audit.md`) and
match their own documented findings. No script references a removed
API, a renamed module, or a superseded scenario.

## Summary

One outdated comment was found and fixed. No dead code, no unauthorized
duplication, no stale scripts, and no TODO markers were found. Two
declared-but-unused dependencies and one unloaded config file were found
and are documented (not removed -- both are reserved for already-planned
future components, not leftovers). One module
(`reasoning/utility.py`) is correct and tested but not wired into any
live decision path, consistent with the manuscript's own framing of that
section as conceptual/unifying rather than a literal pipeline stage.
