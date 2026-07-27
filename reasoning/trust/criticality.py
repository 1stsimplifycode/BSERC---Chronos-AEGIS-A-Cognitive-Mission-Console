"""The Mission Criticality Model (manuscript Section 7.4).

Formalizes the explicit link between a mission's own operational importance
and how the trust threshold and evaluation criteria applied to it are set.

    tau(m) = alpha_base + (1 - alpha_base) * kappa_m

so that a higher-criticality mission (kappa_m closer to 1) requires a higher
mission-trust value T(m,t) before the scheduler (Section 14) will recommend
proceeding, all else equal.

    U(m) = kappa_m * P_success(m,t) - (1 - kappa_m) * D(m,t)
           - CMCS-weighted cyber exposure (Section 12.6.3)

This mission-criticality-adjusted utility is a *special case* of Section
4.2's general objective U(sigma(m,t)): setting alpha=eta=0 and folding all
criticality weighting into beta=kappa_m, delta_w=(1-kappa_m), with gamma
absorbed into the CMCS-weighted exposure term, recovers exactly the U(m)
above. Both forms are retained deliberately (see reasoning/utility.py for
the general form): the general objective is the theoretical anchor every
CMI mechanism is framed against, while this specialized form is what the
scheduler (Section 14.1, not yet implemented) actually evaluates.

Limitation (stated explicitly in Section 7.4): kappa_m is presently a design
parameter requiring domain-expert elicitation or calibration against
historical mission-criticality judgments, rather than a quantity learned
end-to-end. This module treats it as a required input, not something it
estimates.

The CMCS-weighted cyber exposure term (Section 12.6.3, Cyber Mission
Criticality Score) is part of Sections 11-13 (Cognitive Cyber Reasoning /
mission-aware cyber metrics), which are NOT YET IMPLEMENTED in this phase
(see cyber/README.md). `mission_criticality_utility` therefore accepts it as
an explicit, caller-supplied `cyber_exposure` argument rather than computing
it internally -- there is no fallback default, so a caller cannot
accidentally treat "cyber exposure not modeled yet" as "cyber exposure is
zero".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionCriticalityModel:
    """tau(m) = alpha_base + (1 - alpha_base) * kappa_m (Section 7.4)."""

    alpha_base: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_base <= 1.0:
            raise ValueError(f"alpha_base must be in [0,1], got {self.alpha_base}")

    def trust_floor(self, kappa_m: float) -> float:
        """tau(m): the criticality-adjusted minimum mission-trust threshold."""
        if not 0.0 <= kappa_m <= 1.0:
            raise ValueError(f"kappa_m must be in [0,1], got {kappa_m}")
        return self.alpha_base + (1.0 - self.alpha_base) * kappa_m

    def mission_criticality_utility(
        self,
        kappa_m: float,
        p_success: float,
        expected_delay: float,
        cyber_exposure: float,
    ) -> float:
        """U(m) = kappa_m*P_success - (1-kappa_m)*D - CMCS-weighted cyber exposure."""
        if not 0.0 <= kappa_m <= 1.0:
            raise ValueError(f"kappa_m must be in [0,1], got {kappa_m}")
        return kappa_m * p_success - (1.0 - kappa_m) * expected_delay - cyber_exposure
