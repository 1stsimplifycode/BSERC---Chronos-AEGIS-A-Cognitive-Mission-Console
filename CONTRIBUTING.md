# Contributing

## Before you start

`paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf` is the authoritative
specification for this repository. Any change to `graph/`, `reasoning/`,
`memory/`, `agents/`, `cyber/`, `scheduler/`, `digital_twin/`,
`evaluation/`, or `benchmarks/` should be traceable to a specific
manuscript section, or clearly marked as infrastructure the manuscript
does not cover (as Phase 7-9's evaluation/benchmark/audit tooling is).
See `docs/developer-guide.md`'s "Repository conventions" for the full
set of standing conventions (manuscript-as-source-of-truth, documented
assumptions, no fabricated results, `Null*` dependency injection, reuse
over duplication).

## Setup

```bash
git clone https://github.com/1stsimplifycode/chronos-aegis.git
cd chronos-aegis
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest
ruff check .
mypy graph reasoning memory agents scheduler cyber digital_twin evaluation benchmarks scripts --ignore-missing-imports
```

All three must be clean. If you touched `digital_twin/`, `evaluation/`,
or `benchmarks/`, also run `python -m scripts.reproduce_all` twice and
confirm `datasets/generated/reproducibility_report.json`'s
`deterministic_content_hash` matches between the two runs -- see
`docs/reproducibility-guide.md`.

## What a good change looks like here

- **Cites the manuscript.** New code's docstring opens with the exact
  manuscript passage (quoted) and section number it implements.
- **Documents its assumptions.** Where the manuscript underspecifies a
  formula, constant, or design choice, say so explicitly with an
  `ASSUMPTION (documented)` note -- see any file matching
  `grep -rl "ASSUMPTION (documented)" --include=*.py .` for the
  established pattern.
- **Never fabricates a result.** A missing prerequisite (a baseline, a
  database, an optional dependency) should produce an honest empty
  result or a raised, caught, and warned-about `NotImplementedError` --
  never a value that looks real but isn't.
- **Has a test that would fail if the change were wrong.** Not just "the
  code runs" -- an analytic formula gets a hand-computed or
  finite-difference check; a determinism claim gets a same-seed-twice
  test (and, if it could plausibly be a cross-process issue, a
  `subprocess`-based test -- see `docs/reproducibility-guide.md` for why
  this matters).
- **Reuses existing mechanisms** (`ingest_telemetry_edge`,
  `NodeHistory.append_*`, `MetaMemory.recompute`,
  `evaluation.batch.run_single`, the `Null*` dependency-injection
  pattern) rather than duplicating state or logic that already exists
  elsewhere -- see `digital_twin/README.md`'s "No duplicated state"
  principle for the canonical example.
- **Documents negative or unfavorable findings, not just favorable
  ones.** If an experiment or benchmark you add produces an unfavorable
  result, it belongs in `docs/negative-results.md`, not adjusted or
  omitted.

## Reporting an issue

Use the repository's issue tracker. For a suspected manuscript-vs-
implementation discrepancy, please cite the manuscript section and the
specific code location; see `docs/manuscript-consistency-report.md` for
the two known manuscript-internal wording inconsistencies already on
file (not implementation bugs) so you don't duplicate them.

## Code of conduct

See `CODE_OF_CONDUCT.md`.
