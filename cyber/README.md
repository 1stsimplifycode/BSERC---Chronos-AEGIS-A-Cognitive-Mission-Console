# cyber/ -- Cognitive Cyber Reasoning, Mission-Aware Cyber Risk Propagation, Temporal Attack Graphs (Sections 11-13)

## Status: implemented

| Section | Component | File |
|---|---|---|
| 11 | Cognitive Cyber Reasoning (mu(q)) | `cognitive_reasoning.py` |
| 11 / 12.2 | Shared similarity-weighted consequence formula | `consequence.py` |
| 12.1 | Base risk propagation (widest-path attenuation) | `risk_propagation.py` |
| 12.2 | AI-Driven Attack Chain Reasoning | `attack_chain.py` |
| 12.3 | Software supply-chain / firmware trust decay | `risk_propagation.py` |
| 12.4 | Zero-Trust Reasoning | `risk_propagation.py` |
| 12.5 | Adversarial AI Robustness | `robustness.py` |
| 12.6.2 | Dependency-subgraph betweenness centrality | `centrality.py` |
| 12.6.1-12.6.3 | MTES, MDRI, CMCS | `metrics.py` |
| 13 | Temporal Attack Graph | `attack_graph.py` |
| 14.1 lines 1-2 | MTES -> CyberExposure bridge for Algorithm 6 | `scheduler_integration.py` |

Every module's docstring cites the exact manuscript passage it implements
and, where the manuscript leaves a detail underspecified, an explicit
"ASSUMPTION (documented)" paragraph explaining the derivation and why it
was chosen -- see `docs/ARCHITECTURE.md` for the consolidated list. No
equation in this package was invented independent of the manuscript.

## Reuse, not a parallel subsystem

Per the manuscript's own repeated framing ("Cognitive Cyber Reasoning is a
**complementary**... mechanism," Section 11; risk propagation "feed[s]
directly into the cyber impact head C(m,t) of Section 7," Section 12.1),
this package does not duplicate machinery that already exists elsewhere:

- **Retrieval** (Sections 11, 12.2, 13) reuses `memory.emm.EpisodicMissionMemory`'s
  Algorithm 3 (`EpisodicMemoryStore.search`, extended in this phase with an
  optional `predicate` filter) exactly, via one shared formula
  (`cyber.consequence.predict_mission_consequence`) both Section 11 and
  Section 12.2 call with different predicates.
- **Attenuation** (Section 12.1, and MTES's `attenuation(v->mission)` term)
  reuses Section 7.2's exact `rho(u->v,t)` formula
  (`reasoning.trust.propagation.propagated_trust_contribution`), extended
  across multiple hops as a widest path rather than an unrelated formula.
- **Dependency identification** (`Dep(m)` for MDRI, and the "intermediate
  dependency" match field for attack chains) reuses the same
  `dependency_identity` node-attribute convention `memory.features` already
  established in Phase 2 -- not a second schema.
- **Feature matching** (Section 13's "a temporal attack graph path is
  itself a candidate feature vector s") reuses `memory.features.FeatureVector`
  and Section 12.2's own `AttackChainReasoning`, extended (not replaced)
  with two new categorical fields (`entry_node_type`, `target_criticality_tier`).
- **Argumentation-level conflict handling** (Section 12.5's sensor-spoofing
  mitigation) reuses `agents.argumentation` (Section 9, Phase 3) unchanged;
  `cyber/robustness.py` does not reimplement conflict detection.
- **Scheduling** (Algorithm 6 lines 1-2) now reads real MTES values via
  `cyber.scheduler_integration.compute_cyber_exposure` instead of a
  caller-supplied placeholder dict (the Phase 6 primary objective).

## Documented assumptions (see each module's docstring for the full derivation)

The manuscript specifies these sections' *structure* precisely but leaves
several closed-form details unspecified; every one of the following is a
documented, overridable design choice, not a fabricated formula:

- **12.1's per-hop attenuation formula**: reuses Section 7.2's `rho(u->v,t)`
  extended to multiple hops as a widest (maximum-product) path.
- **11's mu(q) severity scale**: `MissionOutcome` is categorical; mu(q)
  averages numeric outcomes, so `cyber.consequence.DEFAULT_CONSEQUENCE_SEVERITY`
  supplies a documented 0.0-1.0 ordinal scale, overridable per call.
- **11/12.2's mu(q)-blend weight**: `mu_weight=0.5` (equal weighting)
  because the manuscript says mu(q) is "weighed against" propagation
  evidence without naming a specific weight.
- **12.6.1's psi(mission_criticality)**: the identity function
  psi(kappa_m) = kappa_m, reusing Section 7.4's established [0,1] scale.
- **12.6.1's time integral**: closed-form under the assumption that the
  widest attenuation path at query time remains the widest path across the
  integration window (documented in `metrics.py`).
- **12.6.2's centrality**: betweenness centrality (Brandes' algorithm),
  the manuscript's own named example.
- **12.6.3's "MDRI-normalized centrality of v within Dep(m)"**: read as
  v's own MDRI contribution as a fraction of the mission's total MDRI.
- **13's mission-phase imminence**: exponential decay in time-until-phase,
  mirroring the decay family already used throughout (Section 6.1,
  `memory.similarity.recency_weight`).
- **13's third attack-chain matching field** (`target_criticality_tier`):
  no TRMG-derivable source exists for it, so it is caller-supplied.

All such parameters (severity scale, blend weights, decay rates,
similarity weights) are exactly the kind Section 12.6's own text flags as
"requiring empirical calibration and validation against the digital
twin's ground truth (Section 15)" -- consistent with the rest of this
repository, nothing here claims calibration has happened.

## What's still not implemented

- **A vulnerability database / SBOM ingestion.** `intrinsic_risk` (rho(v))
  is caller-supplied throughout this package; there is no CVE/CVSS feed or
  SBOM parser to derive it from automatically.
- **Causal discovery** for the TRMG's causal-graph subset (Section 10.4)
  remains explicitly out of scope, as it was in Phase 4.
- **The Mission Digital Twin's sensor-spoofing scenario** (Section 12.5,
  the other half of that threat surface's mitigation, alongside
  cross-agent argumentation which *is* implemented via reuse) requires
  Section 15, out of scope for this phase.
- **Model-poisoning corroboration** (`cyber.robustness.is_corroborated`)
  is implemented and tested but still not called from
  `EpisodicMissionMemory.update` or from `reasoning.continual.algorithm7.mission_adapt`
  (Section 16's continual-learning loop, implemented in Phase 8): Phase 8's
  scope for Algorithm 7 was the theta update, Fisher recompute, and
  credibility/Meta-Memory reuse Section 16.5 names explicitly -- it does
  not name a corroboration-gated memory-write check, so none was added.
  Wiring `is_corroborated` into the write path a poisoned episode would
  take remains a real, open integration gap, not a scope exclusion
  anymore now that Section 16 itself is implemented.

## Usage

```python
from graph.trmg.model import TemporalReasoningMemoryGraph
from cyber.metrics import mission_dependency_risk_index

# Betweenness centrality (12.6.2's own named example) needs >= 3 dependency
# nodes to be non-trivial -- see scripts/experiment_rq12_cyber_metrics.py's
# _demonstrate_nonzero_mdri for the full account of why a 1- or 2-node
# example would print 0.0 even with real intrinsic_risk supplied.
graph = TemporalReasoningMemoryGraph()
graph.add_node("mission")
graph.add_node("satcom", dependency_identity="satcom_module@1.4")
graph.add_node("satcom_firmware", dependency_identity="satcom_fw@2.0")
graph.add_node("vendor_component", dependency_identity="vendor_component@1.0")
graph.add_edge("satcom", "mission", timestamp=0.0, confidence=0.8, decay_rate=0.05)
graph.add_edge("satcom_firmware", "satcom", timestamp=0.0, confidence=0.9, decay_rate=0.05)
graph.add_edge("vendor_component", "satcom_firmware", timestamp=0.0, confidence=0.9, decay_rate=0.05)

result = mission_dependency_risk_index(
    graph, "mission", intrinsic_risk={"satcom_firmware": 0.6}, k_hop=6
)   # Section 12.6.2
print(result.value, result.dependency_nodes)   # 0.6 {'satcom', 'satcom_firmware', 'vendor_component'}
```

See `examples/run_ttrn_pipeline_with_scheduler.py` for the full capstone
demonstration (every component in this README composed into one Algorithm
1 query, with the Explainability Pipeline showing real cyber content in
its Evidence-, Graph-, and Scheduler-levels) and `tests/cyber/` (80+
tests: unit, hand-computed validation examples for MTES/MDRI/CMCS and
betweenness centrality, and end-to-end pipeline/regression tests) for
verification.
