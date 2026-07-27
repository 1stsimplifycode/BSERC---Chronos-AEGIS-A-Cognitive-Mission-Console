# configs/

`default.yaml` holds the numeric parameters the manuscript fixes or
requires calibration for: the k-hop bound and agent count (Section 5),
default TRMG decay/priority values (Section 6.1), the Mission Criticality
Model's `alpha_base` (Section 7.4, explicitly flagged there as "requires
calibration"), the Mission Utility weights (Section 4.2), and the Bayesian
layer's default prior variance (Section 7.3).

Nothing in `reasoning/` or `graph/` reads this file automatically yet --
today's tests and `examples/run_ttrn_pipeline.py` pass parameters directly
to constructors. Wiring config loading into a future `backend/` service is
Phase 2 work.
