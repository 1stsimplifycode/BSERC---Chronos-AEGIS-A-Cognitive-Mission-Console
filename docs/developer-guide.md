# Developer Guide

## Setup

```bash
pip install -e ".[dev]"
```

Optional dependencies (neither required -- see each module's own
docstring for how their absence is handled honestly, never silently):

```bash
pip install scipy         # real paired-t-test/Wilcoxon p-values (evaluation.statistics)
pip install matplotlib    # figures (digital_twin.visualization)
```

## Quality gates (run after every change)

```bash
pytest                                                                              # 548 tests, all passing
ruff check .
mypy graph reasoning memory agents scheduler cyber digital_twin evaluation benchmarks scripts --ignore-missing-imports
```

All three must be clean before considering a change complete -- this is
the standard this repository has held itself to since Phase 1.

## Repository conventions

- **Manuscript is the source of truth.** `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf`.
  Every module's docstring cites the specific section/algorithm it
  implements. Read `docs/ARCHITECTURE.md` before touching a component you
  haven't worked in before.
- **Document every assumption.** Where the manuscript underspecifies a
  formula, constant, or design choice, the code says so explicitly with
  an `ASSUMPTION (documented)` note, not a silent default.
- **Never fabricate a result.** No module reports a number it did not
  actually compute. Missing prerequisites (a baseline, a database, an
  optional dependency) produce an honest `None`/empty result or a raised
  `NotImplementedError`/`ImportError`, caught and surfaced as a warning,
  never papered over.
- **Reuse, don't duplicate.** New phases wire into existing mechanisms
  (`ingest_telemetry_edge`, `NodeHistory.append_*`, `MetaMemory.recompute`,
  `evaluation.batch.run_single`, ...) rather than building parallel state
  or parallel computation paths. See `digital_twin/README.md`'s "no
  duplicated state" principle for the canonical example.
- **`Null*` dependency injection.** Optional CMI components
  (`NullEpisodicMemory`, `NullArgumentationReconciler`, `NullCausalEngine`,
  `NullPatchScheduler`) are the pattern for "not configured" -- extend it
  rather than adding ad hoc `if component is None` branches.

## Common extension points

**Add a Digital Twin scenario** (`digital_twin/scenarios.py`): compose
existing event types from `digital_twin/events.py`; only add a new event
type if no existing composition can faithfully represent it, and document
the composition choice in the scenario's own docstring (see
`build_sensor_spoofing`'s docstring for the pattern). Add it to
`ALL_SCENARIO_BUILDERS`; leave `CANONICAL_SCENARIO_BUILDERS` (the original
ten, Section 15.1) untouched.

**Add a benchmark axis** (`benchmarks/`): one module per axis, a frozen
dataclass result type, reuse `evaluation.batch`/`evaluation.aggregate`
wherever the axis is already close to what they compute. Wire it into
`benchmarks.report.run_benchmark_suite` and `BenchmarkSuiteReport`.

**Add a continual-learning benchmark scenario sequence**
(`benchmarks/convergence.py`): pass a different `scenario_builders` tuple
to `continual_learning_convergence` -- order matters (it's a real mission
sequence, not a random sample).

## Testing conventions

- One test file per source module, same relative path under `tests/`.
- Analytic gradients get a finite-difference check
  (`tests/reasoning/continual/test_objective.py` is the reference
  example).
- Determinism claims get an actual same-seed-twice test; cross-process
  determinism claims get a `subprocess`-based test (see
  `tests/scripts/test_reproduce_all.py`) -- same-process tests cannot
  catch `PYTHONHASHSEED`-class bugs (see `docs/reproducibility-guide.md`).
- Optional-dependency code paths get `importlib.util.find_spec(...)`-gated
  tests that skip (not fail) when the dependency isn't installed, and
  still test the "dependency absent" behavior unconditionally.
