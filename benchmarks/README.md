# benchmarks/ -- Benchmark Suite (Phase 8)

**Status: implemented, as a self-benchmarking suite -- not as baseline
comparisons.**

Section 17.2 names the baselines each research question is evaluated
against: memory networks, abstract argumentation, structural causal
discovery, continual-learning baselines, CVSS-based scoring, and
conventional MITRE ATT&CK-style attack-graph reasoning (see Table 0,
Section 2). Implementing any of those baselines is out of scope for this
package -- see `evaluation/README.md` for the full account of what is and
is not implemented toward each RQ.

What this package does implement, for real, against the actually-built
system: repeated deterministic-seed runs, scalability studies (TRMG
neighborhood depth, Episodic Memory size), latency and memory-usage
measurement, trust evolution and cyber-metric aggregation (reusing
`evaluation/`), continual-learning convergence (Section 16, RQ10),
retrieval accuracy (RQ6/RQ7), and scheduler-calibration quality --
exported as one structured `BenchmarkSuiteReport` (JSON).

```python
from digital_twin.scenarios import ALL_SCENARIO_BUILDERS
from benchmarks import run_benchmark_suite, write_report_json

report = run_benchmark_suite(ALL_SCENARIO_BUILDERS, seeds=tuple(range(10)))
write_report_json(report, "benchmark_report.json")
```

See `docs/benchmark-guide.md` for what each axis measures and why, and
`python -m scripts.reproduce_all` for running this as part of the full
reproducibility workflow.
