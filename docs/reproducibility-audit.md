# Reproducibility Audit (Phase 9)

This document is the step-by-step account of what a fresh clone of this
repository needs to do to install, test, generate datasets, execute
experiments, and regenerate figures/tables/benchmark outputs -- and the
real numbers observed running that sequence during this audit.

## Environment specification

| Item | Audited value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 (the workflow is pure-Python and has no OS-specific code path; Linux/macOS are expected to work identically, not separately verified in this audit) |
| Python | 3.11.9 (pyproject.toml requires \>=3.11) |
| CPU | Intel64 Family 6 Model 142 Stepping 10 (a standard laptop-class x86-64 CPU; nothing in this repository requires a GPU, a specific CPU feature, or more than one core) |
| RAM | Not a constraint observed -- peak traced allocation during a full benchmark suite run is on the order of tens of kilobytes per scenario run (`tracemalloc`-measured, see `benchmark_report.json`'s `peak_memory_bytes` fields); a few hundred MB of headroom for the Python interpreter itself is sufficient |
| Disk | See "Disk requirements" below |

## Installation

```bash
git clone https://github.com/1stsimplifycode/chronos-aegis.git
cd chronos-aegis
pip install -e ".[dev]"
```

`pyproject.toml`'s `[project.dependencies]` (numpy, pydantic, PyYAML) and
`[project.optional-dependencies].dev` (pytest, pytest-cov, mypy, ruff,
black) are the only two dependency lists needed for everything in this
audit. For an exact reproduction of the audited environment, use
`requirements-lock.txt` instead (see that file's own header for the two-
step install it needs, since it does not include this project's own
`-e .`).

Optional, for the two code paths documented as untested in this audit
(`docs/negative-results.md`, Section 5):

```bash
pip install scipy matplotlib
```

## Step-by-step: audited commands and observed results

All commands below were run from the repository root, in order, in the
audited environment (no step depends on manual intervention or GUI
interaction).

### 1. Run the test suite

```bash
pytest
```

**Observed**: `548 passed, 5 skipped in ~3.1-3.6s`. The 5 skips are
matplotlib-gated (`tests/digital_twin/test_visualization.py`) and are
expected, not failures -- see that test file's own skip reasons.

### 2. Run the quality gates

```bash
ruff check .
mypy graph reasoning memory agents scheduler cyber digital_twin evaluation benchmarks scripts --ignore-missing-imports
```

**Observed**: `All checks passed!` (ruff); `Success: no issues found in 86 source files` (mypy).

### 3. Generate datasets, run experiments, run benchmarks, regenerate tables/figures -- the one-command workflow

```bash
python -m scripts.reproduce_all
```

**Observed**: completes in 7-17 seconds wall-clock (two audited runs;
variance is normal system-load/disk-cache variance, not a determinism
problem -- see below). Produces:

```
datasets/generated/
  rq12_cyber_metrics_summary.json
  rq6_rq7_memory_retrieval_precision.json
  benchmark_report.json
  ablation_report.json
  reproducibility_report.json
  publication/summary_tables.md
  publication/trust_evolution.png       (only if matplotlib is installed)
  publication/metric_dashboard.png      (only if matplotlib is installed)
```

In the audited environment (matplotlib not installed), the two `.png`
files were not produced and the script printed
`figures skipped: matplotlib is not installed (optional dependency)`
rather than failing -- this is the documented, honest behavior, not an
error.

### 4. Verify reproducibility

Run step 3 twice (two separate `python` process invocations) and compare
`datasets/generated/reproducibility_report.json`'s
`deterministic_content_hash` between the two runs.

**Observed**: both audited runs produced the identical hash
`b031c30c3c96c5b2005ff0e7b8aec3985a57947f742707f61a69045649124071`
for `seeds: [0..9]`, `scenario_count: 18` -- confirmed reproducible
across two fresh process invocations with no special environment
variables set (this specifically exercises the Phase 8 fix to a real
`PYTHONHASHSEED`-related cross-process determinism bug; see
`docs/reproducibility-guide.md`).

### 5. Run individual experiment scripts (optional, subset of step 3)

```bash
python -m scripts.experiment_rq12_cyber_metrics
python -m scripts.experiment_rq6_rq7_memory_retrieval
```

**Observed**: both run standalone in \<1 second each and reproduce the
same findings documented in `evaluation/README.md` and
`docs/negative-results.md`.

## Deterministic seed documentation

| Component | Seed source | Default |
|---|---|---|
| `scripts/reproduce_all.py` | module-level `SEEDS = tuple(range(10))` | seeds 0-9 |
| `scripts/experiment_rq12_cyber_metrics.py` | module-level `SEEDS = tuple(range(10))` | seeds 0-9 |
| `scripts/experiment_rq6_rq7_memory_retrieval.py` | module-level `SEEDS = tuple(range(10))` | seeds 0-9 |
| `digital_twin.twin.MissionDigitalTwin(mission_id, seed=...)` | caller-supplied, defaults to `0` | per-call |
| `reasoning.trust.gnn.MissionAwareGNN(..., seed=...)` | caller-supplied, defaults to `0` | per-call |
| `benchmarks.*` functions | caller-supplied `seed`/`seeds` parameter, no hidden default | required argument |

Every seeded component derives its randomness from `numpy.random.default_rng(seed)`
or (for per-node feature vectors) `hashlib.sha256`-derived integer seeds
-- neither is affected by `PYTHONHASHSEED` (see
`docs/reproducibility-guide.md` for the one place that used to be, and
the Phase 8 fix).

## Expected runtime

| Command | Observed wall-clock |
|---|---|
| `pytest` | ~3.1-3.6 s |
| `ruff check .` | \<1 s |
| `mypy ...` (86 files) | a few seconds |
| `python -m scripts.reproduce_all` (10 seeds, 18 scenarios) | 7-17 s |
| `python -m scripts.experiment_rq12_cyber_metrics` alone | \<1 s |
| `python -m scripts.experiment_rq6_rq7_memory_retrieval` alone | \<1 s |

No command in this repository's reproducibility workflow takes more than
about 20 seconds on ordinary laptop-class hardware.

## Disk requirements

| Item | Observed size |
|---|---|
| Repository checkout (source + `.git`) | a few MB |
| `.venv` (full dev install: numpy, pydantic, PyYAML, pytest, mypy, ruff, black + transitive deps) | ~197 MB |
| `datasets/generated/` after a full `reproduce_all` run | ~108 KB (JSON + Markdown; figures add a few hundred KB more if matplotlib is installed) |

Well under 1 GB total for a complete, working environment plus every
generated artifact.

## Optional dependencies

| Package | Enables | Behavior when absent |
|---|---|---|
| `scipy` | Real paired t-test / Wilcoxon signed-rank p-values in `evaluation.statistics.paired_comparison` | Returns `p_value_t=None`, `p_value_wilcoxon=None` with an explanatory `note`; the t-statistic and Cohen's d (stdlib-only) are still computed |
| `matplotlib` | Figures in `digital_twin.visualization` and `scripts/reproduce_all.py`'s publication assets | Raises a clear `ImportError` if a plotting function is called directly; `scripts/reproduce_all.py` catches this and prints `figures skipped: matplotlib is not installed (optional dependency)` rather than failing the whole workflow |

Neither package is a hard dependency anywhere in this repository; their
absence never silently degrades a numeric result into a fabricated one
(see `docs/negative-results.md`, Section 5, for what specifically remains
untested without them).

## Reproducibility checklist

- [x] Fresh install (`pip install -e ".[dev]"`) succeeds from `pyproject.toml` alone
- [x] `pytest` passes with no manual configuration
- [x] `ruff check .` and `mypy ...` are clean
- [x] `python -m scripts.reproduce_all` regenerates every dataset, table, and (given matplotlib) figure with no manual intervention
- [x] Two independent runs of the same workflow with the same seeds produce an identical `deterministic_content_hash`, verified across separate process invocations (not just within one Python session)
- [x] Every optional-dependency code path fails honestly (clear error or documented `None`/skip) rather than silently, when the dependency is absent
- [x] No step requires GPU, network access, or manual data download
- [x] No step requires more than ~200 MB disk or more than ~20 seconds wall-clock

This checklist was executed in full during this audit, in the order
above, in one continuous session, with no step skipped or assumed.
