# scheduler/ -- AI Patch Scheduler (Section 14, Algorithm 6)

## Status: implemented

`AIPatchScheduler` (`algorithm6.py`) is the real `PatchSchedulerInterface`
implementation: the feasibility check (delay is infeasible when
`t_patch > t_window`, forcing `patch_now` regardless of cost) and the
weighted-cost argmin over `{patch_now, delay}` (Algorithm 6, lines 5-10).
`NullPatchScheduler` (`interfaces.py`) is the explicit no-scheduler opt-out.

Section 14 is titled specifically "AI Patch Scheduler" -- Algorithm 6's
scope is exactly the patch-timing decision, not a general-purpose mission
timeline scheduler (no broader scheduler is separately specified elsewhere
in the manuscript).

Wired into `reasoning/ttrn/pipeline.py`'s `TTRNPipeline` at Algorithm 1
lines 11-13: pass `scheduler=AIPatchScheduler()` plus a
`PatchSchedulingRequest` to `infer()`'s `patch_scheduling_request` argument
to enable it for a given query; omitting the request skips lines 12-13
silently (an ordinary, expected outcome, not a missing-component gap).

## Where its two cost terms come from

Algorithm 6 lines 1-4 specify *how* to compute its two cost terms; this
module itself (`algorithm6.py`) only implements lines 5-10 (the decision
procedure), taking both terms as `PatchSchedulingRequest.cyber_exposure` /
`.mission_trust_loss` inputs -- but as of Phase 6, neither is a
placeholder:

- **CyberExposure(a)** (lines 1-2): "may now be computed via the MTES
  metric of Section 12.6.1" -- `cyber.scheduler_integration.compute_cyber_exposure`
  does exactly that: CyberExposure(delay) is MTES integrated over the full
  `t_window`, CyberExposure(patch_now) is MTES integrated over the shorter
  `t_patch` (the residual exposure while the patch is being applied).
- **MissionTrustLoss(a)** (lines 3-4): "delta_T under do(patch->applied)...
  via Section 10.5" -- computed via `reasoning.causal.algorithm5.StructuralCausalEngine`,
  the real Section 10.5 causal engine (Phase 4).

Neither this module nor its callers hardcode either value.
`examples/run_ttrn_pipeline_with_scheduler.py` reproduces the manuscript's
own worked scenario (Section 14.1) -- a mission starting in 12 minutes, a
patch requiring 4 minutes, and a communication window closing in 6
minutes -- with both cost terms computed for real, to whichever action the
weighted argmin actually selects (the manuscript's own recommended answer,
"delay," falls out of the real computation rather than being assumed).

`tests/scheduler/` covers the feasibility branch, the argmin branch
(including weight sensitivity), request validation, and
`NullPatchScheduler`; `tests/cyber/test_scheduler_integration.py` and
`tests/cyber/test_pipeline_integration.py` cover the real CyberExposure
bridge and the full Algorithm-1-through-Algorithm-6 pipeline,
respectively, including a regression assertion that no
"backend not configured" warnings are produced when every component is
wired in.

## Scheduler Learning (Phase 8): calibrating w_cyber, w_trust

`scheduler/calibration.py` adds evidence-driven calibration of Algorithm
6's two weights (w1, w2 in Section 14's own notation) -- **Algorithm 6
itself (`algorithm6.py`) is completely unmodified**: `calibrate_weights`
only ever *produces* new `weight_cyber`/`weight_trust` values for a
caller's *next* `PatchSchedulingRequest`, preserving "the optimization
formulation must remain intact; only learned parameters may evolve."

The calibration rule (a documented choice, not a manuscript-specified
formula -- Section 16 does not name the scheduler among the parameters
its continual objective covers) nudges each weight by the Pearson
correlation, over accumulated `SchedulingOutcomeRecord`s, between that
term's own cost value at decision time and the outcome it is named to
anticipate (CyberExposure vs. `CYBER_INCIDENT`; MissionTrustLoss vs.
`FAILURE`/`DELAY`), then renormalizes so `weight_cyber + weight_trust`
stays constant -- only their relative balance evolves. See
`benchmarks/scheduler_quality.py` for a benchmark verifying this rule
responds correctly to a known, controllable correlation strength in
generated data, and `tests/scheduler/test_calibration.py` for unit
coverage including the no-signal (constant cost values) and
both-weights-start-at-zero edge cases.
