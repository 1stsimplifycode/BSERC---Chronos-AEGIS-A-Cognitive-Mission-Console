# reasoning/causal/ -- Causal Mission Intelligence (Section 10)

## Status: implemented

| Section | Component | File |
|---|---|---|
| 10.1 | The Counterfactual Engine (retained, correlational) | `counterfactual.py` |
| 10.2 | Structural Causal Model M=(U,V,F) | `scm.py` |
| 10.5 | Algorithm 5: Causal Counterfactual Generation | `algorithm5.py` |
| 10.7 | Eight-Level Explainability Pipeline | `explainability.py` |

`StructuralCausalEngine` (`algorithm5.py`) is the real `CausalEngineInterface`
implementation, dispatching each candidate intervention `do(X=x)` to
whichever branch Algorithm 5 specifies: the causal branch (SCM mutilation +
forward evaluation, exact given a fully specified M) if `X` is a registered
SCM variable, otherwise the correlational fallback (re-running the
mission-aware GNN over an intervened TRMG copy, Section 10.1), explicitly
labeled `"correlational, not yet verified causal"` so a caller is never
misled about which kind of claim they're looking at.

See `examples/run_ttrn_pipeline_with_causal.py` for real Episodic Mission
Memory + Structured AI Argumentation + Causal Mission Intelligence + the
Explainability Pipeline running together over one Algorithm 1 query, and
`examples/run_ttrn_pipeline_with_scheduler.py` for the full capstone
adding the AI Patch Scheduler (Section 14) on top, with a populated
Scheduler-level explanation. `tests/reasoning/causal/` covers the SCM
(against hand-computed linear examples), both counterfactual branches,
Algorithm 5's dispatch logic, and all eight explanation levels.

## Documented scope limitations (see each module's docstring for detail)

- **Causal discovery is not implemented.** Section 10.4 states this is "an
  open problem, not a solved one" for the manuscript itself, deferred to
  Section 17's evaluation protocol once the Mission Digital Twin exists.
  `StructuralCausalModel` is populated from domain-supplied structural
  functions (`register()`), never inferred from data.
- **Only interventional (do) queries are implemented**, not the full
  observational/interventional/counterfactual three-way distinction of
  Section 10.3. Recovering `P(T|do(X))` from purely observational data
  would require Pearl's abduction step over a distribution on the
  exogenous variables U, which the manuscript does not specify; Algorithm
  5, the only algorithmic specification in Section 10, only ever calls for
  the interventional computation, which `do()` implements exactly (graph
  mutilation) given a fully specified M.
- **The correlational fallback branch only intervenes on TRMG edge
  confidence** (via a `"source->target"` variable-naming convention, see
  `algorithm5.parse_edge_variable`), not the more abstract mission-specific
  variables named in Section 10.3.1's worked examples (`patch_status`,
  `R_comm`, `navigation_confidence`, ...).
- **The Explainability Pipeline does not compute anything new**; it
  assembles the eight levels from outputs the caller already has.
  Scheduler-level (Section 14.1) is populated for real when a
  `PatchSchedulingResult` is supplied, and reported unavailable only when
  patch scheduling simply wasn't requested for that query. As of Phase 6,
  Evidence-level and Graph-level also surface real cyber reasoning content
  when supplied -- Cognitive Cyber Reasoning's mu(q) and its supporting
  episodes (Section 11, `ExplanationInputs.cyber_consequence`), and risk
  propagation attenuation plus the MDRI dependency chain (Sections 12.1,
  12.6.2, `ExplanationInputs.cyber_exposure` / `.dependency_risk`) --
  additively, without adding a ninth level (Section 10.7 fixes exactly
  eight) or changing the other six levels' behavior. See `cyber/README.md`.
