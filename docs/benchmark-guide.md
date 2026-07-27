# Benchmark Guide (Phase 8)

`benchmarks/` benchmarks the real, implemented system against itself --
it does **not** implement any of Section 17.2's named RQ baselines
(AutoGen/CrewAI-style dialogue, CVSS scoring, MITRE ATT&CK-style attack
graphs, ...); see `evaluation/README.md` for why those remain out of
scope. `benchmarks/README.md`'s "not yet implemented" status refers
specifically to those baseline comparisons, which this package does not
add.

## Running the full suite

```python
from digital_twin.scenarios import ALL_SCENARIO_BUILDERS
from benchmarks import run_benchmark_suite, write_report_json

report = run_benchmark_suite(ALL_SCENARIO_BUILDERS, seeds=tuple(range(10)))
write_report_json(report, "benchmark_report.json")
```

Or via the one-command workflow: `python -m scripts.reproduce_all` (see
`docs/reproducibility-guide.md`).

## What each axis measures

| Module | Axis | Real mechanism |
|---|---|---|
| `benchmarks.scalability.k_hop_scalability` | TRMG neighborhood depth (Section 17.4) | Reuses `evaluation.batch.run_single`'s `k_hop` parameter directly |
| `benchmarks.scalability.memory_size_scalability` | Episodic Memory size \|M_e\| (Section 8.5, 17.4) | Pre-populates a shared `EpisodicMissionMemory` with real episodes before timing one further query |
| `benchmarks.convergence.continual_learning_convergence` | Continual-learning convergence (Section 16, RQ10) | Runs a real mission sequence through Algorithm 1 + Algorithm 7, recording actual task loss / EWC penalty per mission |
| `benchmarks.retrieval.retrieval_precision_at_1` | Retrieval accuracy (RQ6/RQ7) | Leave-one-out precision@1 over real stored episodes, vs. majority-class baseline |
| `benchmarks.scheduler_quality.scheduler_calibration_benchmark` | Scheduler calibration | Runs `scheduler.calibration.calibrate_weights` against synthetic-but-structured outcome histories with a controllable, known correlation strength, and verifies the weights move the expected direction |
| `evaluation.batch` + `evaluation.aggregate` (via `run_benchmark_suite`) | Repeated runs, deterministic seeds, latency, memory usage, trust evolution, cyber metrics | Already-implemented Phase 7 infrastructure, reused unchanged |

Agent count `\|A\|` scaling (also named in Section 17.4) is not
benchmarked: Section 9.1's nine domain-specialized agents are fixed by
the manuscript (`agents/roster.py`), and nothing in this repository
varies `\|A\|` at runtime.

## `scheduler_quality`'s synthetic data, and why that's not fabrication

`scheduler_calibration_benchmark` generates outcome histories with a
*controllable, known* correlation strength between cost terms and
outcomes, then verifies `calibrate_weights` (a real, deterministic
algorithm, `scheduler/calibration.py`) responds to that signal correctly.
This tests the calibration *mechanism*, not a claim about real-world
scheduler decision quality (which would need the RQ5 baseline named
above, not implemented here) -- the synthetic input is documented as
exactly that in the module's own docstring, and the benchmark reports
what the real algorithm computed against it, not an assumed result.

## Every other axis uses only real, executed Digital Twin data

Scalability, convergence, and retrieval all run actual
`digital_twin.integration.run_scenario_with_pipeline` calls -- no number
is estimated, interpolated, or fabricated. `tests/benchmarks/` verifies
determinism (same seed -> identical result) for every axis.
