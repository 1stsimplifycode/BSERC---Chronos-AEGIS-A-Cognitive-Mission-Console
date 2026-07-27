"""Mission Utility: the unifying optimization objective (manuscript Section 4.2).

    U(sigma(m,t)) = alpha*T_post(m,t) + beta*P_success(m,t) - gamma*C(m,t)
                    - delta_w*D(m,t) - eta*Upsilon(m,t)

where T_post and P_success are read from sigma(m,t), C(m,t) is the cyber
impact head (Section 7.1, extended by Sections 11-12's MTES/MDRI/CMCS),
D(m,t) is expected mission delay, Upsilon(m,t) is posterior uncertainty, and
alpha,beta,gamma,delta_w,eta >= 0 are mission-dependent weights.

Every mission-facing decision is then posed as

    delta(m,t) = argmax_{a in Actions} U(Phi(sigma(m,t-1), e_t) | do(a))
                 subject to T_post(m,t) >= tau(m)

i.e. select the action that maximizes mission utility under the resulting
state, subject to the criticality-adjusted trust floor tau(m) of Section 7.4.

The manuscript notes this general objective is what Section 7.4's Mission
Criticality Model specializes for criticality weighting (alpha=eta=0,
beta=kappa_m, delta_w=1-kappa_m); see reasoning/trust/criticality.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True)
class MissionUtilityWeights:
    """alpha, beta, gamma, delta_w, eta >= 0 (Section 4.2)."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta_w: float = 1.0
    eta: float = 1.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta_w", "eta"):
            if getattr(self, name) < 0:
                raise ValueError(f"weight '{name}' must be >= 0")


@dataclass(frozen=True)
class MissionUtilityInputs:
    """The four quantities U reads from sigma(m,t) (Section 4.2)."""

    trust_posterior: float
    p_success: float
    cyber_impact: float
    expected_delay: float
    uncertainty: float


def mission_utility(
    inputs: MissionUtilityInputs, weights: MissionUtilityWeights | None = None
) -> float:
    """U(sigma(m,t)) as defined in Section 4.2."""
    weights = weights or MissionUtilityWeights()
    return (
        weights.alpha * inputs.trust_posterior
        + weights.beta * inputs.p_success
        - weights.gamma * inputs.cyber_impact
        - weights.delta_w * inputs.expected_delay
        - weights.eta * inputs.uncertainty
    )


Action = TypeVar("Action")


@dataclass(frozen=True)
class ActionEvaluation(Generic[Action]):
    action: Action
    inputs: MissionUtilityInputs
    utility: float
    feasible: bool  # whether T_post(m,t) >= tau(m) held for this action


def select_best_action(
    candidates: dict[Action, MissionUtilityInputs],
    trust_floor: float,
    weights: MissionUtilityWeights | None = None,
) -> ActionEvaluation[Action] | None:
    """delta(m,t) = argmax_{a in Actions} U(...) subject to T_post(m,t) >= tau(m).

    `candidates` maps each candidate action to the MissionUtilityInputs that
    would result from applying it (i.e. the caller has already computed
    Phi(sigma(m,t-1), e_t) | do(a) for each a -- how `do(a)` is applied is
    Section 10's Causal Mission Intelligence, not yet implemented; this
    function only performs the constrained argmax over already-evaluated
    candidates).

    Returns the best *feasible* action (trust_posterior >= trust_floor), or
    None if no candidate satisfies the trust floor -- an explicit,
    non-fabricated "no safe action" result rather than silently returning an
    infeasible best-by-utility action.
    """
    weights = weights or MissionUtilityWeights()
    evaluations = [
        ActionEvaluation(
            action=action,
            inputs=inputs,
            utility=mission_utility(inputs, weights),
            feasible=inputs.trust_posterior >= trust_floor,
        )
        for action, inputs in candidates.items()
    ]
    feasible = [e for e in evaluations if e.feasible]
    if not feasible:
        return None
    return max(feasible, key=lambda e: e.utility)
