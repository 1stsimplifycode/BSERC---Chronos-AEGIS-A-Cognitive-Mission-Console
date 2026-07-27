# Release Notes

## Current state: pre-1.0, feature-complete through Sections 1-16, audited

`pyproject.toml` currently declares `version = "0.1.0"`. No version has
been tagged in git yet (`git log` shows commits through Phase 6 only;
Phases 7-9 are the current working-tree state, uncommitted). This
document describes what a `v1.0.0` release would contain if cut from the
current state; it is written now, ahead of that tag, so the release
description is prepared rather than improvised at tag time.

## What `v1.0.0` would contain

A complete, tested, documented, audited implementation of Chronos-AEGIS's
Cognitive Mission Intelligence framework as formalized in
`paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf`, Sections 1-16:

- The Temporal Reasoning Memory Graph (Section 6), with historical
  snapshots and reversible temporal pruning.
- Episodic Mission Memory (Section 8), with non-destructive,
  outcome-driven Episodic Learning (Phase 8).
- Structured AI Argumentation (Section 9).
- Causal Mission Intelligence (Section 10), including the eight-level
  Explainability Pipeline.
- Cognitive Cyber Reasoning, Mission-Aware Cyber Risk Propagation, and
  Temporal Attack Graphs (Sections 11-13), including MTES/MDRI/CMCS.
- The AI Patch Scheduler (Section 14), with evidence-driven weight
  calibration (Phase 8).
- The Mission Digital Twin (Section 15): 18 scenarios (10 canonical +
  1 baseline + 2 composites + 5 Phase 8 firmware scenarios), a
  discrete-event engine, and deterministic replay.
- Adaptive Learning and Continual Reasoning (Section 16): Algorithm 7
  and the Regularized Continual Objective.
- An evaluation framework, a benchmark suite, a reproducibility
  workflow (`scripts/reproduce_all.py`), and a full research-artifact
  audit (`paper/artifact/`, `docs/traceability-matrix.md`).

**548 tests, ruff-clean, mypy-clean, 91% line coverage.**

## What `v1.0.0` would explicitly not claim

No research question (RQ1-RQ13) has a full baseline comparison executed
-- no baseline is implemented for any of them. Section 16.6's
catastrophic-forgetting question and Section 10.4's causal discovery
remain open, exactly as the manuscript itself frames them. See
`docs/traceability-matrix.md` for the complete, honest, per-claim
account, and `docs/negative-results.md` for every unfavorable finding
preserved rather than hidden.

## Known, disclosed limitations at this snapshot

- `pydantic` and `PyYAML` are declared dependencies not yet imported by
  any source file (reserved for the not-yet-built `backend/` and a
  not-yet-built config-loader respectively; see
  `docs/code-quality-audit.md`).
- `scipy` and `matplotlib` are optional; neither was installed when this
  snapshot's numbers were generated, so the real-p-value and figure-
  generation code paths were not exercised in that run (they are tested
  and fail/degrade honestly when absent -- see
  `docs/reproducibility-guide.md`).
- `backend/`, `frontend/`, `deployment/`, `docker/` remain unimplemented
  stubs -- no production API, dashboard, or deployment exists yet.

## Upgrade notes

Not applicable -- no prior tagged release exists.

## Acknowledgements

Developed against `paper/Chronos-AEGIS_CMI_Manuscript_v4.pdf` as the
sole source of truth throughout every phase of development; see
`CITATION.cff` for citation metadata.
