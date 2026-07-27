# agents/ -- Multi-Agent Hypothesis Generation & Structured AI Argumentation (Section 9)

## Status

- `roster.py`: the nine domain-specialized agent identities Section 9.1
  names (Aircraft Engineer, Cyber Analyst, Satellite Planner, Weather
  Analyst, Mission Planner, Maintenance Engineer, Communication Specialist,
  Supply Chain Analyst, Safety Officer) -- `|A| = 9` per Section 5.3.
- `hypothesis.py`: **implemented**. `AgentHypothesis` (claim / evidence /
  confidence / counterarguments, Section 9.2) and
  `HypothesisGeneratorInterface`. The concrete domain reasoning behind each
  of the nine roles is **not implemented** -- the manuscript names each
  agent's role but does not specify the reasoning logic that turns "its own
  slice of the TRMG" into a claim, so this repository does not invent one.
- `argumentation/`: **implemented**. `StructuredArgumentAggregator`
  (`aggregation.py`) is the real Algorithm 4: `ArgumentationGraph`
  (`graph.py`, Section 9.2), Dung-style grounded/preferred extension
  semantics (`semantics.py`), and `Conflict(H)`/consensus (`consensus.py`,
  Section 9.3) -- including the escalation path (Algorithm 4, lines 6-8)
  for genuinely unresolvable disagreement (e.g. an odd attack cycle), which
  raises rather than silently defaults. `NullArgumentationReconciler`
  (`interfaces.py`) remains as an explicit no-argumentation opt-out.

See `examples/run_ttrn_pipeline_with_argumentation.py` for a runnable
demonstration of both a cleanly resolvable attack and an escalated one.
`tests/agents/` covers the graph construction, semantics (against
hand-worked classical Dung AF examples), consensus arithmetic, and the
full Algorithm 4 aggregator.

## Usage

```python
from agents.hypothesis import AgentHypothesis
from agents.argumentation.aggregation import StructuredArgumentAggregator
from memory.meta import MetaMemory

hypotheses = (
    AgentHypothesis(agent_id="aircraft_engineer", claim="delay departure",
                     evidence=("engine_health",), confidence=0.8),
    AgentHypothesis(agent_id="mission_planner", claim="proceed as scheduled",
                     evidence=("schedule",), confidence=0.6),
)
aggregator = StructuredArgumentAggregator(meta_memory=MetaMemory())
decision = aggregator.aggregate(hypotheses)   # Algorithm 4, Section 9.4
print(decision.claim, decision.escalated, decision.conflict_score)
```

Pass `argumentation=aggregator` to `reasoning.ttrn.pipeline.TTRNPipeline`
to wire this into a full Algorithm 1 run (Algorithm 1, line 10).

## Documented simplifications (see each module's docstring for detail)

- **Preferred-extension tie-break**: when multiple preferred extensions
  exist, `StructuredArgumentAggregator` picks the one whose resulting
  decision scores highest under Section 9.3's own argmax criterion, then
  breaks further ties by extension size and sorted agent-id -- the
  manuscript does not specify a tie-break.
- **Default credibility (0.5)**: applied to any agent Meta Memory has no
  calibration record for yet; Section 9.6 states credibility is "learned
  from historical outcomes" but does not specify a value before any
  history exists.
- **`counterarguments` as agent_ids, not claim text**: Section 9.2 defines
  `counter` as "the set of other agents' claims that claim attacks";
  `agents/hypothesis.py` operationalizes this as the set of attacked
  `agent_id`s (a stable per-round identifier) rather than matching on raw
  claim strings.

## What's still not connected

`decide()`'s winning claim and `StructuredArgumentAggregator`'s output feed
into `reasoning/ttrn/pipeline.py`'s `TTRNResult.reconciled_decision` and
`CognitiveMissionState.argument_graph`. The AI Patch Scheduler (`scheduler/`,
Section 14) is now implemented too, but reads its own `PatchSchedulingRequest`
rather than `ReconciledDecision` directly -- see
`reasoning/ttrn/pipeline.py`'s module docstring for why the two are kept as
separate `TTRNResult` fields rather than one coordinate as the manuscript's
notation suggests.
