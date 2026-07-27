"""Algorithm 7: MissionAdapt(mission_k, o_k) (manuscript Section 16.5).

    1: EMM-Update(mission_k, t, delta, T, E, C*)   // Algorithm 2, records o_k once observed
    2: Update TRMG: apply decay delta to all edges; ingest any new dependency/telemetry
       edges surfaced by mission_k
    3: Draw a replay batch R from M_e (Section 16.3)
    4: theta <- theta - alpha * grad_theta L(theta; mission_k union R)   // regularized update
       (Section 16.2)
    5: Recompute Fisher-information estimate F_i for the updated theta   // for the next call's
       regularization term
    6: Update agent credibility weights w_i (Section 9.3) from o_k vs. each agent's prior claim
    7: Update Meta Memory M_m with the new calibration and credibility state
    8: return updated theta, w, M_m

Line 1 is satisfied by the caller, not re-executed here: `TTRNPipeline.infer`
(Algorithm 1, `reasoning/ttrn/pipeline.py`) already performs EMM-Update
(Algorithm 1 line 15 IS Algorithm 2/Algorithm 7 line 1 -- the same call,
described from two algorithms' point of view, not two separate writes) --
`mission_adapt` below requires that the caller has already invoked
`pipeline.infer(..., outcome=..., embedding=...)` for the concluding
mission before calling this function, and reads that just-appended episode
back out of `memory.episodic` rather than writing it a second time, which
would double-append the same mission's episode.

Lines 6-7 reuse `memory.meta.MetaMemory.recompute` directly (the same
"recompute per-agent calibration and credibility from a full pass over
Episodic Memory" operation Algorithm 2 line 9 and batched consolidation
already use) rather than a second, parallel credibility-update mechanism:
Algorithm 7 line 6's "update... from o_k vs. each agent's prior claim" and
line 7's "update M_m with the new calibration and credibility state" are,
read together, exactly what one `MetaMemory.recompute` call already does.

HONEST FINDING (discovered by running `benchmarks.convergence`, not
assumed in advance): with `mission_adapt` called once per mission -- the
literal reading of Algorithm 7, whose line 4 is a single gradient step,
not an inner training loop -- the EWC penalty term's *gradient*
contribution to every update after the first is exactly zero. This is a
real mathematical consequence, not a bug: `state.reference_heads` (theta*)
is set, every call, to the theta this same call just produced (line 8's
"return updated theta" becomes the next call's theta*), so at the start of
the *next* call, `heads == reference_heads` exactly -- and both the EWC
penalty's value and its gradient (`2*lambda*F*(heads-reference)`) are
identically zero whenever `heads == reference_heads`. The regularization
term therefore has no effect on any single-step-per-mission update under
this literal reading of Algorithm 7; it would only matter if a caller took
more than one gradient step against the same `state.reference_heads`
before calling `mission_adapt` again (Algorithm 7 does not specify doing
so). Section 16.6 already flags that "we have not evaluated whether the
specific regularization strength lambda and replay-batch composition
prevent catastrophic forgetting in practice" -- this is a concrete,
measured instance of exactly that open question, not resolved here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.emm import EpisodicMissionMemory
from memory.meta import AgentCalibration
from reasoning.continual.objective import (
    HEAD_NAMES,
    HeadWeights,
    ReplayItem,
    estimate_fisher_diagonal,
    outcome_targets,
    regularized_update,
)
from reasoning.trust.gnn import MissionAwareGNN


@dataclass
class ContinualLearningState:
    """theta* (the anchor point Section 16.2's penalty measures distance
    from) and F (the diagonal Fisher-information estimate), carried by the
    caller across successive `mission_adapt` calls -- this is exactly the
    persistent state Algorithm 7 line 8 returns ("updated theta") for the
    *next* call's regularization term, kept explicit here rather than
    hidden as mutable state on `MissionAwareGNN` itself.
    """

    reference_heads: HeadWeights
    fisher: HeadWeights


def initial_state(gnn: MissionAwareGNN) -> ContinualLearningState:
    """theta* = theta_0 (the freshly constructed, untrained heads) and
    F = 0 (no prior-mission importance information exists yet -- an honest
    "unknown," not a fabricated estimate) -- the state to pass into the
    first ever `mission_adapt` call for a given `gnn` instance."""
    reference = gnn.get_head_weights()
    return ContinualLearningState(
        reference_heads=reference,
        fisher={name: np.zeros_like(w) for name, w in reference.items()},
    )


@dataclass(frozen=True)
class MissionAdaptResult:
    """Algorithm 7 line 8's return value: updated theta (already applied
    to `gnn` in place, and also returned for inspection), the state to
    pass into the *next* mission_adapt call, and everything computed along
    the way, for logging -- nothing here is fabricated after the fact."""

    updated_heads: HeadWeights
    next_state: ContinualLearningState
    task_loss: float
    ewc_penalty: float
    replay_batch: tuple[str, ...]
    """mission_id of every replay episode actually drawn (mission_k's own
    id is not included -- see mission_adapt's docstring)."""
    credibility: dict[str, AgentCalibration]


def mission_adapt(
    graph: TemporalReasoningMemoryGraph,
    gnn: MissionAwareGNN,
    memory: EpisodicMissionMemory,
    state: ContinualLearningState,
    *,
    now: float,
    ewc_lambda: float = 0.1,
    learning_rate: float = 0.01,
    replay_batch_size: int = 8,
    rng: np.random.Generator | None = None,
) -> MissionAdaptResult:
    """Algorithm 7 lines 2-8 (line 1 already satisfied by the caller --
    see module docstring). Requires `memory.episodic` to be non-empty and
    its most recent entry to carry a non-empty `embedding` (i.e. the
    caller's `pipeline.infer(...)` call that concluded mission_k must have
    been given `embedding=...`); raises ValueError otherwise rather than
    silently skipping the parameter update.
    """
    entries = memory.episodic.all_entries()
    if not entries:
        raise ValueError(
            "mission_adapt requires memory.episodic to contain at least one "
            "episode (the just-concluded mission_k); call "
            "pipeline.infer(..., outcome=..., embedding=...) first"
        )
    mission_k_entry = entries[-1]
    if not mission_k_entry.embedding:
        raise ValueError(
            f"the most recent episode (mission_id={mission_k_entry.mission_id!r}) has no "
            "recorded embedding; pass embedding=... to TTRNPipeline.infer() so "
            "mission_adapt has an input to differentiate the prediction heads against"
        )

    # Line 2: TRMG update -- decay, reusing Algorithm 1 line 1's own
    # mechanism (graph/trmg/model.py's apply_decay). "Ingest any new
    # dependency/telemetry edges surfaced by mission_k" is the caller's
    # responsibility via the same graph.ingest_telemetry_edge Algorithm 1
    # already uses -- there is no separate telemetry channel for this
    # function to invent.
    graph.apply_decay(now)

    # Line 3: draw a replay batch R from M_e (Section 16.3), excluding
    # mission_k's own entry (already included as the batch's first item
    # below) and any entry with no recorded embedding (not eligible for
    # replay -- see EpisodicEntry.embedding's own docstring).
    replayable = [e for e in entries[:-1] if e.embedding]
    generator = rng or np.random.default_rng()
    if replayable:
        sample_size = min(replay_batch_size, len(replayable))
        indices = generator.choice(len(replayable), size=sample_size, replace=False)
        replay_entries = [replayable[i] for i in indices]
    else:
        replay_entries = []

    batch = [
        ReplayItem(
            embedding=np.asarray(mission_k_entry.embedding, dtype=float),
            targets=outcome_targets(mission_k_entry.outcome),
        )
    ] + [
        ReplayItem(
            embedding=np.asarray(entry.embedding, dtype=float),
            targets=outcome_targets(entry.outcome),
        )
        for entry in replay_entries
    ]

    # Line 4: theta <- theta - alpha * grad_theta L(theta; mission_k union R).
    new_heads, task_loss, ewc_penalty = regularized_update(
        gnn,
        batch,
        state.reference_heads,
        state.fisher,
        ewc_lambda=ewc_lambda,
        learning_rate=learning_rate,
    )
    gnn.set_head_weights(new_heads)

    # Line 5: recompute F_i for the updated theta, over the same batch.
    new_fisher = estimate_fisher_diagonal(new_heads, batch)

    # Lines 6-7: agent credibility + Meta Memory, via the same recompute
    # operation Algorithm 2 line 9 already defines (see module docstring).
    memory.meta.recompute(memory.episodic.all_entries())

    next_state = ContinualLearningState(reference_heads=new_heads, fisher=new_fisher)

    # Line 8: return updated theta, w, M_m.
    return MissionAdaptResult(
        updated_heads=new_heads,
        next_state=next_state,
        task_loss=task_loss,
        ewc_penalty=ewc_penalty,
        replay_batch=tuple(e.mission_id for e in replay_entries),
        credibility=memory.meta.all_calibrations(),
    )


__all__ = [
    "HEAD_NAMES",
    "ContinualLearningState",
    "MissionAdaptResult",
    "initial_state",
    "mission_adapt",
]
