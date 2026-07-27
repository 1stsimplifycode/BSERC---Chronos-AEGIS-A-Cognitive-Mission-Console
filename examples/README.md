# examples/

- `run_ttrn_pipeline.py`: runs Algorithm 1 (TTRN-Infer, Section 5.2)
  end-to-end over a small hand-built Temporal Reasoning Memory Graph, using
  every default `Null*` collaborator (no memory, argumentation, or causal
  backend), and prints which pipeline stages it had to skip and why.
- `run_ttrn_pipeline_with_memory.py`: the same pipeline, but with a real
  `EpisodicMissionMemory` (Section 8) wired in, across three missions that
  share a vulnerable-dependency/topology pattern -- demonstrates EMM-Update
  persisting episodes, EMM-Retrieve finding them on later missions, and
  batched consolidation distilling a semantic rule.
- `run_ttrn_pipeline_with_argumentation.py`: the same pipeline with a real
  `StructuredArgumentAggregator` (Section 9, Algorithm 4) wired in --
  demonstrates a cleanly resolvable one-directional attack, and a
  three-way mutual-attack cycle that Algorithm 4 correctly escalates to a
  human operator rather than silently defaulting.
- `run_ttrn_pipeline_with_causal.py`: real Episodic Mission Memory,
  Structured AI Argumentation, and a `StructuralCausalEngine` (Section 10,
  Algorithm 5) all wired into one pipeline run, showing both Algorithm 5's
  causal branch (a domain-supplied Structural Causal Model) and its
  correlational fallback branch, then assembling the eight-level
  Explainability Pipeline (Section 10.7) over the result.
- `run_ttrn_pipeline_with_scheduler.py`: the full capstone -- adds a real
  `AIPatchScheduler` (Section 14, Algorithm 6) on top of everything above.
  **Both** of Algorithm 6's cost terms are computed for real: MissionTrustLoss
  via the causal engine (lines 3-4) and CyberExposure via Section 12.6.1's
  MTES (`cyber.scheduler_integration`, lines 1-2) -- no placeholder cost
  terms remain anywhere in the scheduling path. Reproduces the
  manuscript's own worked Section 14.1 scenario (a mission starting in 12
  minutes, a 4-minute patch, a 6-minute communication window) -- computed,
  not hardcoded, to whichever action the weighted argmin actually selects
  (it lands on "delay," matching the manuscript's own stated recommendation,
  as a consequence of the real computation, not an assumption). Prints all
  eight explanation levels, including real Cognitive Cyber Reasoning
  evidence and risk-propagation/dependency-chain content under Evidence-
  and Graph-level, and the real scheduling decision under Scheduler-level.

```
pip install -e ".[dev]"
python -m examples.run_ttrn_pipeline
python -m examples.run_ttrn_pipeline_with_memory
python -m examples.run_ttrn_pipeline_with_argumentation
python -m examples.run_ttrn_pipeline_with_causal
python -m examples.run_ttrn_pipeline_with_scheduler
```
