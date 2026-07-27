"""Section 16.2: the Regularized Continual Objective.

    L(theta) = L_task(theta; mission_k) + lambda * Sum_i F_i * (theta_i - theta*_i)^2

where `L_task` is the task loss on the newly completed mission k, `theta*`
are the parameters after training on missions 1..k-1, `F_i` is a diagonal
Fisher-information estimate of parameter theta_i's importance to
previously learned missions, and `lambda` controls the elastic-weight-
consolidation-style penalty for moving parameters important to past
missions (Section 16.2's stability-plasticity trade-off).

Scope (documented, see reasoning/trust/gnn.py's own Phase 8 note): theta
here is exactly `MissionAwareGNN`'s six prediction-head weight vectors
(`get_head_weights`/`set_head_weights`), not the embedding/attention layer
(`_feature_projection`), which this module leaves untouched. The
manuscript names "the mission-aware GNN's weights" generically (Section
16.1) without enumerating which specific parameters theta denotes; the six
regression heads are the natural, minimal, and only genuinely supervised
subset (they are literally Section 7.1's "prediction heads"), so they are
the scope chosen here.

ASSUMPTION (documented): the manuscript gives L_task's *name* and role but
not its closed form. This module defines it as the sum, across the six
heads, of squared error between each head's predicted output
(`MissionAwareGNN.predict`, already squashed through its head-specific
sigmoid/softplus) and a per-head target derived deterministically from the
mission's realized `MissionOutcome` (`outcome_targets` below). This is a
standard, simple, fully-specified choice for a scalar "task loss" where
none is given -- not a claim that this is *the* loss the manuscript
intends, since none is stated.

ASSUMPTION (documented): the diagonal Fisher-information estimate F_i uses
the empirical-Fisher approximation F_i ~= (dL_task/dtheta_i)^2 (the
squared task-loss gradient), a standard, widely-used closed form for
"a diagonal Fisher-information estimate" (Section 16.2's own phrasing,
"for example, a diagonal Fisher-information estimate," explicitly leaves
the specific estimator open).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from memory.outcome import MissionOutcome
from reasoning.trust.gnn import MissionAwareGNN

HeadWeights = dict[str, np.ndarray]

HEAD_NAMES: tuple[str, ...] = (
    "trust",
    "p_success",
    "expected_delay",
    "cyber_impact",
    "comm_reliability",
    "operational_confidence",
)


def outcome_targets(outcome: MissionOutcome) -> dict[str, float]:
    """The per-head supervised target derived from a realized
    `MissionOutcome` (documented ASSUMPTION -- see module docstring).

    SUCCESS: high trust/p_success/comm_reliability/operational_confidence,
    no delay, no cyber impact.
    FAILURE: low across every head, no delay or cyber-impact signal
    specifically implicated (a failure with no further detail could be
    anything).
    DELAY: neutral trust/p_success/comm_reliability/operational_confidence
    (0.5 -- "not clearly a success or failure, but late"), high delay, no
    cyber impact.
    CYBER_INCIDENT: low trust/p_success/comm_reliability/operational_confidence,
    no delay signal, high cyber impact.
    OTHER: neutral (0.5) across every head -- an outcome with no specific
    directional signal is honestly represented as "no informative signal,"
    not defaulted to any of the four specific categories above.
    """
    if outcome == MissionOutcome.SUCCESS:
        return {
            "trust": 1.0,
            "p_success": 1.0,
            "expected_delay": 0.0,
            "cyber_impact": 0.0,
            "comm_reliability": 1.0,
            "operational_confidence": 1.0,
        }
    if outcome == MissionOutcome.FAILURE:
        return {
            "trust": 0.0,
            "p_success": 0.0,
            "expected_delay": 0.0,
            "cyber_impact": 0.0,
            "comm_reliability": 0.0,
            "operational_confidence": 0.0,
        }
    if outcome == MissionOutcome.DELAY:
        return {
            "trust": 0.5,
            "p_success": 0.5,
            "expected_delay": 1.0,
            "cyber_impact": 0.0,
            "comm_reliability": 0.5,
            "operational_confidence": 0.5,
        }
    if outcome == MissionOutcome.CYBER_INCIDENT:
        return {
            "trust": 0.0,
            "p_success": 0.0,
            "expected_delay": 0.0,
            "cyber_impact": 1.0,
            "comm_reliability": 0.0,
            "operational_confidence": 0.0,
        }
    # MissionOutcome.OTHER: no directional signal to encode honestly.
    return dict.fromkeys(HEAD_NAMES, 0.5)


@dataclass(frozen=True)
class ReplayItem:
    """One training example for the continual objective: a stored
    embedding h_m(t) (Section 7.1) and the target derived from its
    episode's own realized outcome."""

    embedding: np.ndarray
    targets: dict[str, float]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _head_prediction_and_local_gradient(name: str, raw: float) -> tuple[float, float]:
    """pred = head's squashed output for raw = embedding . w_name, and
    d(pred)/d(raw) -- the local derivative of that head's own squashing
    function, needed by the chain rule below. Mirrors
    `MissionAwareGNN.predict`'s squashing choices exactly (sigmoid for
    every head except `expected_delay`, softplus for `expected_delay`,
    whose derivative is itself a sigmoid)."""
    if name == "expected_delay":
        pred = math.log1p(math.exp(-abs(raw))) + max(raw, 0.0)  # softplus(raw)
        return pred, _sigmoid(raw)
    pred = _sigmoid(raw)
    return pred, pred * (1.0 - pred)


def task_loss_and_gradients(
    heads: HeadWeights, item: ReplayItem
) -> tuple[float, HeadWeights]:
    """L_task and its gradient wrt every head's weight vector, for one
    `ReplayItem`. Loss is the sum, across the six heads, of squared error
    between that head's prediction and its target (documented ASSUMPTION,
    see module docstring)."""
    loss = 0.0
    gradients: HeadWeights = {}
    for name in HEAD_NAMES:
        w = heads[name]
        raw = float(item.embedding @ w)
        pred, local_grad = _head_prediction_and_local_gradient(name, raw)
        error = pred - item.targets[name]
        loss += 0.5 * error * error
        gradients[name] = (error * local_grad) * item.embedding
    return loss, gradients


def batch_task_loss_and_gradients(
    heads: HeadWeights, batch: Sequence[ReplayItem]
) -> tuple[float, HeadWeights]:
    """L_task(theta; mission_k union R) (Section 16.3): the mean loss and
    mean gradient over `batch` (mission_k's own item plus a replay sample
    R). Raises ValueError on an empty batch -- there is no honest "task
    loss" over zero examples.
    """
    if not batch:
        raise ValueError("batch_task_loss_and_gradients requires at least one item")

    total_loss = 0.0
    total_grad: HeadWeights = {name: np.zeros_like(heads[name]) for name in HEAD_NAMES}
    for item in batch:
        loss, grad = task_loss_and_gradients(heads, item)
        total_loss += loss
        for name in HEAD_NAMES:
            total_grad[name] = total_grad[name] + grad[name]

    n = len(batch)
    mean_grad = {name: total_grad[name] / n for name in HEAD_NAMES}
    return total_loss / n, mean_grad


def ewc_penalty_and_gradients(
    heads: HeadWeights,
    reference_heads: HeadWeights,
    fisher: HeadWeights,
    ewc_lambda: float,
) -> tuple[float, HeadWeights]:
    """The EWC penalty term lambda * Sum_i F_i * (theta_i - theta*_i)^2
    and its gradient 2*lambda*F_i*(theta_i - theta*_i), evaluated at the
    current `heads` against the anchor `reference_heads` (theta*) and
    `fisher` (F)."""
    penalty = 0.0
    gradients: HeadWeights = {}
    for name in HEAD_NAMES:
        diff = heads[name] - reference_heads[name]
        penalty += float(np.sum(fisher[name] * diff * diff))
        gradients[name] = 2.0 * ewc_lambda * fisher[name] * diff
    return ewc_lambda * penalty, gradients


def estimate_fisher_diagonal(heads: HeadWeights, batch: Sequence[ReplayItem]) -> HeadWeights:
    """F_i ~= (dL_task/dtheta_i)^2 (documented empirical-Fisher-diagonal
    approximation, see module docstring), evaluated at `heads` over
    `batch`."""
    _, grad = batch_task_loss_and_gradients(heads, batch)
    return {name: grad[name] ** 2 for name in HEAD_NAMES}


def regularized_update(
    gnn: MissionAwareGNN,
    batch: Sequence[ReplayItem],
    reference_heads: HeadWeights,
    fisher: HeadWeights,
    *,
    ewc_lambda: float,
    learning_rate: float,
) -> tuple[HeadWeights, float, float]:
    """One step of theta <- theta - alpha * grad_theta L(theta; mission_k union R)
    (Algorithm 7, line 4): task loss + EWC penalty, differentiated wrt
    `gnn`'s current head weights. Does not mutate `gnn` -- returns the new
    theta for the caller to apply via `gnn.set_head_weights`, plus the
    task loss and penalty values actually computed (for logging /
    inspection, never fabricated).
    """
    heads = gnn.get_head_weights()
    task_loss, task_grad = batch_task_loss_and_gradients(heads, batch)
    penalty, penalty_grad = ewc_penalty_and_gradients(heads, reference_heads, fisher, ewc_lambda)

    new_heads: HeadWeights = {}
    for name in HEAD_NAMES:
        total_grad = task_grad[name] + penalty_grad[name]
        new_heads[name] = heads[name] - learning_rate * total_grad
    return new_heads, task_loss, penalty
