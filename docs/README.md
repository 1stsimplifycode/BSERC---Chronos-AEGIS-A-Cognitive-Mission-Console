# docs/

- `ARCHITECTURE.md`: the paper-to-repository map -- every manuscript
  component, where it lives in this repo, and its real implementation
  status. Start here.
- `developer-guide.md`: environment setup, running the test suite and
  quality gates, and how to extend the codebase (add a scenario, add an
  event type, add a continual-learning benchmark axis).
- `continual-learning-guide.md`: Section 16 (Algorithm 7, the Regularized
  Continual Objective) -- what is implemented, its scope decisions, and
  the honest findings discovered while implementing and benchmarking it.
- `benchmark-guide.md`: the `benchmarks/` package -- what each axis
  measures and how to run the full suite.
- `reproducibility-guide.md`: `scripts/reproduce_all.py`, what "identical
  output across runs" means here, and the real cross-process determinism
  bug this phase found and fixed.

Not yet written (gated on the components they would document actually
existing): installation/deployment guide, API reference, mathematical
appendix, contributing guide, system/sequence/deployment diagrams.
