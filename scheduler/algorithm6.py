"""Algorithm 6: AI Patch Scheduling (manuscript Section 14.1).

Algorithm 6: AIPatchScheduler(delta, t_start, t_patch, t_window)
    1: Compute CyberExposure(delay) <- MTES integrated over [t, t_window] (Section 12.6.1)
    2: Compute CyberExposure(patch_now) <- residual exposure during the
       t_patch application window
    3: Compute MissionTrustLoss(patch_now) <- delta_T under do(patch->applied)
       evaluated immediately (Section 10.5)
    4: Compute MissionTrustLoss(delay) <- delta_T under do(patch->applied)
       evaluated after t_window
    5: if t_patch > t_window then
    6:     action* <- patch_now // delay is infeasible regardless of cost comparison
    7: else
    8:     action* <- argmin_a [ w_cyber*CyberExposure(a) + w_trust*MissionTrustLoss(a) ]
    9: end if
    10: return action*, with CyberExposure and MissionTrustLoss values attached as evidence

Lines 1-4 (computing the two cost terms) are the caller's responsibility
via `PatchSchedulingRequest.cyber_exposure` / `.mission_trust_loss`
(scheduler/interfaces.py) -- this module implements exactly lines 5-10,
the decision procedure itself, which is what Section 14.1 actually
specifies as Algorithm 6's logic ("given" the two cost terms).
"""

from __future__ import annotations

from typing import Any

from scheduler.interfaces import (
    ACTIONS,
    DELAY,
    PATCH_NOW,
    PatchSchedulingRequest,
    PatchSchedulingResult,
)


class AIPatchScheduler:
    """The real PatchSchedulerInterface implementation: Algorithm 6."""

    def schedule(
        self, reconciled_decision: Any, request: PatchSchedulingRequest
    ) -> PatchSchedulingResult:
        # Lines 5-6: delay is infeasible regardless of cost comparison.
        if request.t_patch > request.t_window:
            return PatchSchedulingResult(
                action=PATCH_NOW,
                feasible=False,
                cost={},
                request=request,
                summary=(
                    f"delay is infeasible (t_patch={request.t_patch} > "
                    f"t_window={request.t_window}); patch_now forced regardless of cost "
                    f"comparison (Algorithm 6, line 6)."
                ),
            )

        # Line 8: weighted argmin.
        cost = {
            action: request.weight_cyber * request.cyber_exposure[action]
            + request.weight_trust * request.mission_trust_loss[action]
            for action in ACTIONS
        }
        action = min(cost, key=lambda a: cost[a])
        return PatchSchedulingResult(
            action=action,
            feasible=True,
            cost=cost,
            request=request,
            summary=(
                f"action={action} selected by weighted-cost argmin over "
                f"{PATCH_NOW}={cost[PATCH_NOW]:.4f}, {DELAY}={cost[DELAY]:.4f} "
                f"(Algorithm 6, line 8)."
            ),
        )
