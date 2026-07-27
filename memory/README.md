# memory/ -- Episodic Mission Memory (Section 8)

**Status: implemented.** `EpisodicMissionMemory` (`memory/emm.py`) is a
real, tested implementation of EMM = (M_w, M_s, M_e, M_m), wired into
`reasoning/ttrn/pipeline.py`'s `TTRNPipeline` via `EpisodicMemoryInterface`
(pass `memory=EpisodicMissionMemory()`; the default remains
`NullEpisodicMemory`, an explicit no-persistence opt-out).

| Section | Component | File |
|---|---|---|
| 8.2.1 | Working Memory M_w | `working.py` |
| 8.2.2 | Semantic Memory M_s + distillation | `semantic.py` |
| 8.2.3 | Episodic Memory M_e + indexed search | `episodic.py` |
| 8.2.4 | Meta Memory M_m (agent calibration) | `meta.py` |
| 8.3 | Algorithm 2 (EMM-Update) | `emm.py` |
| 8.3.1 | Batched consolidation | `consolidation.py` |
| 8.4 | Algorithm 3 (EMM-Retrieve) | `emm.py` |
| 8.4 | sim(q, s_i), recency weighting | `similarity.py` |
| Algorithm 2/3 line 2/1 | ExtractFeatureVector | `features.py` |
| 8.2.3 | Mission outcome categories | `outcome.py` |

See `examples/run_ttrn_pipeline_with_memory.py` for a runnable
end-to-end demonstration: three missions sharing a vulnerable-dependency
and topology pattern, each retrieving the prior ones' episodes, with
consolidation distilling a semantic rule once the pattern has recurred
`theta_pattern` times with a consistent outcome. `tests/memory/` covers
every component individually plus the composed `EpisodicMissionMemory`.

## Documented simplifications (see each module's docstring for detail)

These are deliberate, stated design choices -- not the "not yet
implemented" placeholder pattern used elsewhere in this repository:

- **Indexing** (`episodic.py`): a hash bucket on the two exact-match
  categorical fields, not the full ANN structure (k-d tree / HNSW) Section
  8.5 names as its reference design. Falls back to a full scan for queries
  the bucket can't narrow.
- **`PendingOutcome(m)`** (Algorithm 2, line 3): supplied explicitly by the
  caller as an `outcome` argument, not looked up internally. The Mission
  Digital Twin (`digital_twin/`, Section 15, Phase 7) is now this
  repository's real source for it -- `digital_twin.integration.run_scenario_with_pipeline`
  passes the twin's own ground-truth `MissionOutcome` once a scenario
  completes -- but the interface still takes it as an explicit argument
  rather than reaching into a specific ground-truth source internally, so
  a caller with any other outcome source (a hand-built example, a future
  production telemetry feed) can supply it the same way.
- **Agent calibration metric** (`meta.py`): a Brier-score component against
  a binary realized-success indicator, since `AgentHypothesis` (Section
  9.1) has no structured predicted-outcome field to grade against
  `MissionOutcome`'s five categories more precisely.
- **Similarity weights and the pruning policy** are exactly what Section
  8.6 flags as open empirical questions (RQ6, RQ7) -- both default to
  unweighted/simplest choices, not fitted constants.

## Phase 8: Episodic Learning (non-destructive, evidence-driven)

`memory/reliability.py` adds `ReliabilityLedger`: when a mission concludes
with a real outcome, every episode that was retrieved to help reason
about it is reinforced (if its own outcome agreed) or weakened (if it
disagreed) via an append-only adjustment log -- never by mutating the
stored `EpisodicEntry.outcome_confidence` itself. `EpisodicMemoryStore.search`
ranks by this *effective* confidence, folding the ledger on top of the
original value at read time. Contradictory episodes are retained, not
discarded; the full adjustment history (who adjusted what, when, why) is
queryable via `ReliabilityLedger.history` -- see that module's docstring.

`EpisodicEntry` also gained an `embedding` field (h_m(t), Section 7.1's
GNN output at the time the episode was recorded) -- not part of Section
8.2.3's tuple, but required for genuine experience replay (Section
16.3): `reasoning.continual.algorithm7.mission_adapt` reads it back to
reconstruct a training example for a past episode without needing to
recompute the whole TRMG state that produced it.

`memory/features.py`'s `extract_feature_vector` also now reads
incoming-edge confidence (chi(e).c, Section 6.1) into
`continuous_features` -- diagnosed and fixed while investigating why
Episodic Memory's default retrieval precision (RQ6/RQ7) initially scored
*below* a majority-class baseline (Phase 7's first measurement: precision@1
= 0.3846 vs. a 0.6154 baseline, over the original 13-scenario library).
Root cause: most Digital Twin scenarios never set any of Section 8.2.3's
six categorical fields, so `continuous_features` being otherwise empty
left similarity-based ranking with no real signal for the majority of
scenario types. `scripts/experiment_rq6_rq7_memory_retrieval.py` now runs
a controlled before/after comparison on identical data (the fix applied
vs. not) rather than comparing across separate runs: over the current,
Phase-8-expanded 18-scenario library, precision@1 goes from 0.5556
(categorical fields only) to 0.7778 (+ incoming-edge confidence), now
above the 0.5000 majority-class baseline. Both the diagnosis and the
measured improvement are real and reproducible -- re-run the script to
verify.

## What's still not connected

The mission-aware GNN's attention mechanism (`reasoning/trust/gnn.py`,
Section 7.1) does not yet attend over retrieved episodic memory: `M_ret` is
correctly populated on `CognitiveMissionState.retrieved_memory`, but
bridging a `FeatureVector` (categorical/continuous domain features) into
the GNN's embedding-space input (`np.ndarray`) needs a feature encoder that
does not exist yet. This is a real, open integration gap, not hidden by
this README -- Phase 8's No-EMM ablation (`evaluation/ablation.py`)
independently re-confirmed it: removing Episodic Memory entirely produces
*zero* measurable difference in any GNN prediction-head metric, precisely
because this bridge does not exist. See `docs/continual-learning-guide.md`.
