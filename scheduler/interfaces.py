"""Interface the AI Patch Scheduler (Section 14) must satisfy.

Algorithm 1, lines 11-13:

    11: if the reconciled decision requires patch scheduling then
    12:     delta(m,t) <- AIPatchScheduler(delta(m,t), t_start, t_patch, t_window)
    13: end if

`scheduler.algorithm6.AIPatchScheduler` is the real Algorithm 6
implementation (Section 14.1); `NullPatchScheduler` below is the explicit
no-scheduler opt-out, analogous to the other `Null*` collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

PATCH_NOW = "patch_now"
DELAY = "delay"
ACTIONS = (PATCH_NOW, DELAY)


@dataclass(frozen=True)
class PatchSchedulingRequest:
    """Inputs Algorithm 6 needs beyond delta(m,t) itself.

    `cyber_exposure` and `mission_trust_loss` are supplied by the caller,
    not computed by this module: Algorithm 6 (Section 14.1) says
    "CyberExposure(a) may now be computed via the MTES metric of Section
    12.6.1" -- MTES is not implemented yet (see cyber/README.md) -- and
    MissionTrustLoss(a) is specified as "T under do(patch->applied)... via
    Section 10.5", i.e. via the causal engine already implemented in
    `reasoning/causal/`. See `examples/run_ttrn_pipeline_with_scheduler.py`
    for a real, non-fabricated way to compute both.
    """

    t_start: float
    t_patch: float
    t_window: float
    cyber_exposure: dict[str, float]
    mission_trust_loss: dict[str, float]
    weight_cyber: float = 1.0
    weight_trust: float = 1.0

    def __post_init__(self) -> None:
        for action in ACTIONS:
            if action not in self.cyber_exposure:
                raise ValueError(f"cyber_exposure is missing required action {action!r}")
            if action not in self.mission_trust_loss:
                raise ValueError(f"mission_trust_loss is missing required action {action!r}")
        if self.weight_cyber < 0 or self.weight_trust < 0:
            raise ValueError("weight_cyber and weight_trust must both be >= 0")


@dataclass(frozen=True)
class PatchSchedulingResult:
    """action*, with CyberExposure and MissionTrustLoss attached as
    evidence (Algorithm 6, line 10)."""

    action: str
    feasible: bool
    """False means delay was infeasible (t_patch > t_window) and patch_now
    was forced regardless of cost comparison (Algorithm 6, line 6)."""
    cost: dict[str, float]
    request: PatchSchedulingRequest
    summary: str


class PatchSchedulerInterface(Protocol):
    """The contract Section 14's AI Patch Scheduler must implement."""

    def schedule(
        self, reconciled_decision: Any, request: PatchSchedulingRequest
    ) -> PatchSchedulingResult:
        """AIPatchScheduler(delta(m,t), t_start, t_patch, t_window) (Algorithm 6)."""
        ...


class NullPatchScheduler:
    """An explicit no-scheduler PatchSchedulerInterface. Unlike memory,
    argumentation, and the causal engine, Algorithm 6 has no trivial
    empty-input case to answer honestly (a scheduling decision is always
    requested with real timing parameters), so this always raises rather
    than return a placeholder result. To get real scheduling, pass
    `scheduler=AIPatchScheduler()` (scheduler/algorithm6.py) to TTRNPipeline.
    """

    def schedule(
        self, reconciled_decision: Any, request: PatchSchedulingRequest
    ) -> PatchSchedulingResult:
        raise NotImplementedError(
            "No AI Patch Scheduler backend is configured; cannot schedule. Pass "
            "scheduler=AIPatchScheduler() (scheduler/algorithm6.py) to TTRNPipeline "
            "to enable patch scheduling."
        )
