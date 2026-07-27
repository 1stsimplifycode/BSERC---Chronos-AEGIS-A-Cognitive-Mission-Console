# reasoning/ -- CMI State, TTRN Pipeline, Trust Layer, Causal Engine, Continual Learning

**Status: implemented.** This package holds Chronos-AEGIS's core
reasoning machinery: the Cognitive Mission State and Mission Utility
(Section 4), Algorithm 1's orchestration (Section 5), the trust layer
(Section 7), Causal Mission Intelligence (Section 10), and Adaptive
Learning / Continual Reasoning (Section 16).

| Subpackage / module | Section | What it is |
|---|---|---|
| `state.py` | 4.1 | `CognitiveMissionState` sigma(m,t) |
| `utility.py` | 4.2 | `U(sigma)`, the unifying optimization objective |
| `ttrn/pipeline.py` | 5.2 | `TTRNPipeline`: Algorithm 1's full 16-line control flow, dependency-injecting memory/argumentation/causal/scheduler |
| `trust/gnn.py` | 7.1 | `MissionAwareGNN`: attention + six prediction heads |
| `trust/propagation.py` | 7.2 | Trust Propagation `rho(u->v,t)` |
| `trust/bayesian.py` | 7.3 | Bayesian posterior update |
| `trust/criticality.py` | 7.4 | Mission Criticality Model |
| `trust/calibration.py` | 17.5 (Phase 8) | Expected Calibration Error + histogram-binning recalibration |
| `causal/` | 10 | Structural Causal Model, Algorithm 5, the Explainability Pipeline -- see `reasoning/causal/README.md` |
| `continual/` | 16 (Phase 8) | Algorithm 7 (MissionAdapt) + the Regularized Continual Objective -- see `docs/continual-learning-guide.md` |

`utility.py` (`U(sigma)`) is implemented and tested against Section 4.2's
formula but is not called by any pipeline stage: Section 4.2's own text
frames it as the conceptual objective every CMI mechanism is judged
against theoretically, not a literal per-query computation Algorithm 1
names as one of its sixteen lines -- see `docs/traceability-matrix.md`'s
Section 4.2 row and `docs/code-quality-audit.md` finding 4.

## Usage

The five `examples/run_ttrn_pipeline*.py` scripts are the canonical,
runnable demonstrations of this package, from the bare pipeline (no
optional collaborators) up to the full capstone (real memory,
argumentation, causal engine, and scheduler together):

```python
from graph.trmg.model import TemporalReasoningMemoryGraph
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline

graph = TemporalReasoningMemoryGraph()
# ... populate graph ...
gnn = MissionAwareGNN(embedding_dim=16, feature_dim=8, seed=0)
pipeline = TTRNPipeline(graph, gnn, k_hop=6)   # Null* collaborators by default

result = pipeline.infer("mission_alpha", now=0.0, node_features={...})
```

For continual learning across a mission sequence, see
`docs/continual-learning-guide.md`'s usage example
(`reasoning.continual.algorithm7.mission_adapt`), and for the causal
engine / explainability pipeline, `reasoning/causal/README.md`.

## Design notes

**Dependency injection throughout.** `TTRNPipeline` depends on
`EpisodicMemoryInterface`, `ArgumentationReconcilerInterface`,
`CausalEngineInterface`, and `PatchSchedulerInterface`, never a concrete
class -- each has a real implementation elsewhere (`memory/`, `agents/argumentation/`,
`reasoning/causal/`, `scheduler/`) and a `Null*` opt-out that returns an
honest empty result or raises `NotImplementedError`, caught and recorded
as a warning rather than fabricating a value. This is what let every
phase since Phase 1 add a real collaborator without changing Algorithm
1's call shape.

**theta is scoped to the six prediction heads, not the embedding layer.**
`reasoning/continual/`'s Section 16.2 objective updates
`MissionAwareGNN.get_head_weights()`/`.set_head_weights()` only --
`_feature_projection` (the attention/embedding layer) is left untrained,
a documented scope decision (see `docs/continual-learning-guide.md`).

**Every formula has a hand-computed test.** Bayesian fusion, trust
propagation, the criticality floor, the EWC objective's analytic
gradient (verified against finite differences to ~1e-11) -- none is
"tested" merely by not crashing; see `docs/manuscript-consistency-report.md`
Section 3 for the full equations-vs-code account.

## Tests

`tests/reasoning/` (state, utility, TTRN pipeline), `tests/trust/` (GNN,
propagation, Bayesian, criticality, calibration),
`tests/reasoning/causal/`, `tests/reasoning/continual/` -- 93-100% line
coverage per module; see `docs/code-quality-audit.md` and each
subpackage's own README for the specific uncovered lines and why.
