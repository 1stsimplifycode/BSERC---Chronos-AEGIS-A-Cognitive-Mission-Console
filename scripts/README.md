# scripts/ -- Operational Scripts

Holds experiment-reproduction scripts that run the Digital Twin
(`digital_twin/`, Section 15) and Evaluation Framework (`evaluation/`,
Section 17) end to end. See `evaluation/README.md` for what these do and
do not claim.

## One-command workflow

`reproduce_all.py` -- regenerates every dataset below, runs the
Benchmark Suite and the No-EMM ablation, and exports publication-ready
tables/figures plus a reproducibility report with a content hash for
verifying two runs produced identical results. See
`docs/reproducibility-guide.md`. Run: `python -m scripts.reproduce_all`

## Experiment scripts

- `experiment_rq12_cyber_metrics.py` -- MDRI (Section 12.6.2) across
  Section 15.1's ten canonical scenarios *and* (Phase 8) the five new
  firmware scenarios side by side, plus a supplementary check proving the
  metric is genuinely non-zero on a hand-built graph. Run:
  `python -m scripts.experiment_rq12_cyber_metrics`
- `experiment_rq6_rq7_memory_retrieval.py` -- a controlled before/after
  comparison of Episodic Memory retrieval precision@1 (Section 8.4) on
  identical stored episodes: the default similarity function's
  categorical-fields-only behavior vs. Phase 8's incoming-edge-confidence
  fix, against a majority-class baseline, over every scenario in the
  library. Run: `python -m scripts.experiment_rq6_rq7_memory_retrieval`

All scripts write their machine-readable JSON output to
`datasets/generated/` (gitignored -- regenerate rather than commit) and
print a human-readable summary, including the actual finding from the
last real run of each (see `evaluation/README.md`, "What has actually
been run").

Setup, database-initialization, and demo-data-loading scripts for
`backend/` are not yet populated -- gated on that component existing.

`examples/run_ttrn_pipeline*.py` remain the closest thing to a minimal
runnable entry point for Algorithm 1 alone, without the Digital Twin:
`python -m examples.run_ttrn_pipeline` from the repo root after installing
with `pip install -e ".[dev]"`.
