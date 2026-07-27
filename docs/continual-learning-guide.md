# Continual Learning Guide (Section 16, Phase 8)

Source of truth: `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf`, Section 16
("AI Contribution 7: Adaptive Learning and Continual Reasoning").
Implementation: `reasoning/continual/`.

## What Section 16 specifies

- **16.1 Motivation**: an explicit online continual-learning loop --
  mission outcome -> memory update (Algorithm 2) -> TRMG update (decay +
  new-edge ingestion, Section 6.1) -> agent calibration update (Section
  9.3's credibility weights, read from Meta Memory) -> trust recalibration
  (Section 7's Bayesian posterior) -> improved future reasoning.
- **16.2 Regularized Continual Objective**: an elastic-weight-
  consolidation-style objective, `L(theta) = L_task(theta; mission_k) +
  lambda * sum_i F_i * (theta_i - theta*_i)^2`, to mitigate catastrophic
  forgetting as the mission-aware GNN's parameters `theta` update after
  each mission.
- **16.3 Experience Replay**: `L_task` is computed over a batch mixing
  `mission_k` with a sample drawn from Episodic Memory `M_e`, not just the
  newest mission alone.
- **16.4 Meta-Learning**: explicitly flagged as "a design possibility, not
  a claim" -- **not implemented here**, per the manuscript's own scope.
- **16.5 Algorithm 7 (MissionAdapt)**: the eight-line procedure this
  module implements exactly -- see `reasoning/continual/algorithm7.py`'s
  module docstring for the line-by-line mapping.
- **16.6 Limitations**: the manuscript itself states regularization
  strength `lambda` and replay-batch composition "have not been evaluated
  ... against catastrophic forgetting in practice" -- `benchmarks.convergence`
  is the measurement tool this phase built for exactly that open question;
  it does not claim to answer it.

## What is implemented, and the scope decisions made

`theta` (the parameters Algorithm 7 updates) is scoped to `MissionAwareGNN`'s
six prediction-head weight vectors (`reasoning/trust/gnn.py`'s
`get_head_weights`/`set_head_weights`), not the embedding/attention layer
(`_feature_projection`). The manuscript names "the mission-aware GNN's
weights" generically; the six regression heads are the only genuinely
supervised subset and the natural, minimal reading.

`L_task` (the task loss) has no closed form in the manuscript. This
implementation uses squared error between each head's prediction and a
target derived deterministically from the mission's realized
`MissionOutcome` (`reasoning/continual/objective.py`'s `outcome_targets`)
-- a documented assumption, not a claim that this is *the* loss the
manuscript intends.

The diagonal Fisher-information estimate uses the standard empirical-
Fisher approximation, `F_i ~= (dL_task/dtheta_i)^2` -- Section 16.2's own
phrasing ("for example, a diagonal Fisher-information estimate") leaves
the specific estimator open.

Algorithm 7 line 1 (EMM-Update) is satisfied by the caller, not
re-executed: `TTRNPipeline.infer(..., outcome=..., embedding=...)`
already performs it (Algorithm 1 line 15 *is* Algorithm 7 line 1 -- the
same call). `mission_adapt` reads the just-appended episode back out of
`memory.episodic` rather than writing it a second time. Lines 6-7 (agent
credibility / Meta Memory update) reuse `memory.meta.MetaMemory.recompute`
directly -- the same operation Algorithm 2 line 9 and batched
consolidation already use -- rather than a second, parallel mechanism.

## Honest findings (discovered, not assumed)

**The EWC penalty term's gradient is always zero in a strict, literal
reading of Algorithm 7.** `mission_adapt` takes exactly one gradient step
per call (Algorithm 7 line 4 is a single step, not an inner training
loop), and `state.reference_heads` (theta\*) is set, every call, to the
theta that same call just produced. So at the start of every call after
the first, `heads == reference_heads` exactly -- and both the penalty's
value and its gradient (`2*lambda*F*(heads-reference)`) are identically
zero at that point. This is a real mathematical consequence of the
single-step-per-mission design, not a bug; it means the regularization
term currently has no effect on any update under this literal reading.
See `reasoning/continual/algorithm7.py`'s own docstring for the full
account, and `benchmarks.convergence` for how this was discovered
(`ewc_penalty` is `0.0` at every point in every convergence run).

**Retrieved episodic memory has no path into the GNN's predictions.**
Found while building the No-EMM ablation (`evaluation/ablation.py`):
`TTRNPipeline.infer` never passes `retrieved_memory_features` to
`MissionAwareGNN.embed` (the parameter exists on `embed`, Section 7.1's
"N_k(m) union M_ret" attention pool, but nothing populates it from
`MemoryRetrieveResult`). Removing Episodic Memory therefore has zero
effect on `trust_mean`, `p_success`, or any other prediction-head metric
-- only on memory-specific bookkeeping (`memory_episodes_retrieved`,
`warning_count`). This is a genuine architectural gap, documented here,
not fixed in this phase (fixing it is a new design decision about how to
convert `RankedEpisodicCandidate` entries into embedding-space vectors,
outside Phase 8's task list).

## Usage

```python
from graph.trmg.model import TemporalReasoningMemoryGraph
from digital_twin.twin import MissionDigitalTwin
from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.scenarios import build_satellite_blackout
from memory.emm import EpisodicMissionMemory
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline
from reasoning.continual.algorithm7 import initial_state, mission_adapt

memory = EpisodicMissionMemory()
gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
state = initial_state(gnn)   # theta* = theta_0, F = 0

spec = build_satellite_blackout()
twin = MissionDigitalTwin(spec.name, seed=0)
pipeline = TTRNPipeline(twin.graph, gnn, k_hop=6, memory=memory)
run_scenario_with_pipeline(twin, spec, pipeline)   # Algorithm 1, incl. EMM-Update

result = mission_adapt(twin.graph, gnn, memory, state, now=twin.env.time)
state = result.next_state   # thread across successive missions
```

See `benchmarks/convergence.py` for running a full sequence of missions
and `tests/reasoning/continual/` for the analytic-gradient verification
(checked against finite differences to machine precision).
