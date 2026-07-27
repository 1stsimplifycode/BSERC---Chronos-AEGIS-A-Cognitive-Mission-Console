# Reproducibility Guide (Phase 8)

## One-command workflow

```bash
python -m scripts.reproduce_all
```

Regenerates every experiment dataset (`scripts/experiment_rq12_cyber_metrics.py`,
`scripts/experiment_rq6_rq7_memory_retrieval.py`), runs the Benchmark
Suite (`benchmarks/`) and the No-EMM ablation (`evaluation/ablation.py`),
then writes publication-ready Markdown summary tables and (if matplotlib
is installed) figures, plus a reproducibility report with a content hash.
Everything is written under `datasets/generated/` (gitignored --
regenerate rather than commit):

```
datasets/generated/
  rq12_cyber_metrics_summary.json
  rq6_rq7_memory_retrieval_precision.json
  benchmark_report.json
  ablation_report.json
  reproducibility_report.json
  publication/
    summary_tables.md
    trust_evolution.png       (only if matplotlib is installed)
    metric_dashboard.png      (only if matplotlib is installed)
```

## What "identical output across runs" means here

Every seed-derived numeric result (scenario ground-truth outcomes, MDRI
values, retrieval precision@1, scheduler-calibration weights, continual-
learning task loss, ...) is fully determined by the fixed seeds
`scripts/reproduce_all.py` uses and is covered by
`reproducibility_report.json`'s `deterministic_content_hash` -- a SHA-256
over `benchmark_report.json` and `ablation_report.json` with
`execution_time_seconds`/`peak_memory_bytes` stripped out first (those are
real wall-clock/memory measurements of *that run's own performance*, not
a function of the seed, and legitimately differ between runs on different
hardware or under different load -- they are excluded from the hash
rather than silently included and then blamed on "non-determinism").

Run the workflow twice and compare the two runs'
`deterministic_content_hash` values -- they must match exactly.

## A real determinism bug this phase found and fixed

While first verifying the hash-match claim above, two separate process
runs with the *same seeds* produced *different* hashes. Two distinct bugs
were found and fixed in the course of tracking this down:

1. **`scripts/reproduce_all.py`'s own stripping helper.**
   `dataclasses.asdict()` preserves tuple-typed dataclass fields as
   tuples (not lists); `_strip_nondeterministic` originally only handled
   `dict` and `list`, so timing fields nested inside a tuple silently
   escaped stripping. Fixed by handling `(list, tuple)` together. See
   `tests/scripts/test_reproduce_all.py::test_strip_nondeterministic_recurses_into_tuples`.

2. **`digital_twin/twin.py`'s `node_feature_vector` (the real bug,
   dating to Phase 7).** It derived a per-node RNG seed via Python's
   built-in `hash((self.seed, node_id))`. `hash()` on a `str` -- directly
   or nested in a tuple -- is salted per *process* by `PYTHONHASHSEED`
   (Python's hash-flooding DoS mitigation, on by default since Python
   3.3): the *same* seed and node id produced a *different* feature
   vector in every separate process invocation, while appearing perfectly
   deterministic within any single process. This is exactly why Phase 7's
   own determinism tests never caught it -- pytest runs an entire test
   session in one process, so `hash()` stayed consistent for the whole
   suite. It was only found in Phase 8 by hashing `reproduce_all`'s
   output across two separate `python` invocations. Fixed by deriving the
   seed via `hashlib.sha256` instead, which is not affected by
   `PYTHONHASHSEED` and gives the same digest for the same input in every
   process, forever. See
   `tests/scripts/test_reproduce_all.py::test_node_feature_vector_is_deterministic_across_separate_processes`
   for a regression test that spawns two real subprocesses and compares
   their output.

Both fixes are verified: `scripts/reproduce_all.py` run twice back to
back (fresh `python` process each time, no special environment variables)
now produces byte-identical `deterministic_content_hash` values.

## Optional dependencies

`reproducibility_report.json`'s `optional_dependencies` field records
whether `scipy` (real paired-t-test/Wilcoxon p-values,
`evaluation.statistics.paired_comparison`) and `matplotlib` (figures,
`digital_twin/visualization.py`) were installed for that run. Neither is
required; their absence never silently degrades a result into a
fabricated one -- see each module's own docstring.
