"""Scheduler Learning (Phase 8): evidence-driven calibration of Algorithm
6's weights w_cyber, w_trust, from accumulated mission outcomes.

Algorithm 6 (Section 14.1, `scheduler/algorithm6.py`) is left completely
unmodified by this module: `AIPatchScheduler.schedule` still computes
`action* = argmin_a [w_cyber*CyberExposure(a) + w_trust*MissionTrustLoss(a)]`
exactly as specified. This module only ever *produces* new values for
`w_cyber`/`w_trust` (`PatchSchedulingRequest.weight_cyber`/`.weight_trust`)
for a caller's *next* `PatchSchedulingRequest` -- it never touches the
argmin itself, so "the optimization formulation must remain intact; only
learned parameters may evolve" holds by construction.

ASSUMPTION (documented): the manuscript specifies Algorithm 6's decision
procedure but not how w_cyber/w_trust should be learned or adjusted over
time -- Section 16 (Continual Learning) does not name the scheduler among
the parameters its regularized objective (Section 16.2) covers either.
This module's calibration rule is therefore this repository's own,
explicitly documented choice, not a manuscript-specified formula: each
weight is nudged by the Pearson correlation, over accumulated
`SchedulingOutcomeRecord`s, between that term's own cost value at decision
time and the corresponding realized bad outcome the term is meant to
anticipate (CyberExposure vs. `MissionOutcome.CYBER_INCIDENT`;
MissionTrustLoss vs. `MissionOutcome.FAILURE`/`MissionOutcome.DELAY`) --
a term whose cost value is empirically predictive of the outcome it is
named for is upweighted; a term that is not predictive in the observed
data is downweighted. The two weights are renormalized after every update
so their sum stays constant, which keeps the argmin's effective scale
stable across calibration steps (only their *relative* balance evolves).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from memory.outcome import MissionOutcome
from scheduler.interfaces import ACTIONS, DELAY, PATCH_NOW


@dataclass(frozen=True)
class SchedulingOutcomeRecord:
    """One historical (decision, realized outcome) pair: the action
    Algorithm 6 actually selected, the two cost terms it computed for
    that action, and what actually happened."""

    action_taken: str
    cyber_exposure: float
    mission_trust_loss: float
    outcome: MissionOutcome

    def __post_init__(self) -> None:
        if self.action_taken not in ACTIONS:
            raise ValueError(f"action_taken must be one of {ACTIONS}, got {self.action_taken!r}")


@dataclass(frozen=True)
class SchedulerCalibrationState:
    """The two learned parameters Algorithm 6 consumes -- nothing else.
    Defaults to Algorithm 6's own stated baseline (w1 = w2 = 1.0,
    Section 14)."""

    weight_cyber: float = 1.0
    weight_trust: float = 1.0

    def __post_init__(self) -> None:
        if self.weight_cyber < 0 or self.weight_trust < 0:
            raise ValueError("weight_cyber and weight_trust must both be >= 0")


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Standard Pearson correlation coefficient. Returns 0.0 (no signal)
    if there are fewer than two observations or either series is
    constant (zero variance), rather than dividing by zero."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def calibrate_weights(
    state: SchedulerCalibrationState,
    records: Sequence[SchedulingOutcomeRecord],
    *,
    learning_rate: float = 0.1,
) -> SchedulerCalibrationState:
    """Produce a new `SchedulerCalibrationState` from `records` (see
    module docstring for the calibration rule). Returns `state` unchanged
    if `records` is empty -- "no evidence yet" is an honest no-op, not an
    arbitrary default weight change.
    """
    if not records:
        return state

    cyber_bad = [1.0 if r.outcome == MissionOutcome.CYBER_INCIDENT else 0.0 for r in records]
    trust_bad = [
        1.0 if r.outcome in (MissionOutcome.FAILURE, MissionOutcome.DELAY) else 0.0
        for r in records
    ]
    cyber_values = [r.cyber_exposure for r in records]
    trust_values = [r.mission_trust_loss for r in records]

    cyber_corr = _pearson_correlation(cyber_values, cyber_bad)
    trust_corr = _pearson_correlation(trust_values, trust_bad)

    new_cyber = state.weight_cyber * math.exp(learning_rate * cyber_corr)
    new_trust = state.weight_trust * math.exp(learning_rate * trust_corr)

    # Renormalize so w_cyber + w_trust stays constant -- only the relative
    # balance between the two terms evolves, not their overall scale.
    total_before = state.weight_cyber + state.weight_trust
    total_after = new_cyber + new_trust
    if total_after == 0.0:
        return state
    scale = total_before / total_after
    return SchedulerCalibrationState(
        weight_cyber=new_cyber * scale, weight_trust=new_trust * scale
    )


__all__ = [
    "PATCH_NOW",
    "DELAY",
    "SchedulingOutcomeRecord",
    "SchedulerCalibrationState",
    "calibrate_weights",
]
