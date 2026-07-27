"""Bridges Section 12.6's MTES metric into Section 14's AI Patch Scheduler
(Algorithm 6, scheduler/algorithm6.py), replacing the placeholder
CyberExposure values Phase 5 left as a caller-supplied dict with the real
metric Algorithm 6 itself names.

Algorithm 6, lines 1-2 (Section 14.1):

    1: Compute CyberExposure(delay) <- MTES integrated over [t, t_window] (Section 12.6.1)
    2: Compute CyberExposure(patch_now) <- residual exposure during the t_patch application window

This module implements exactly those two lines: `t_patch` and `t_window`
are the same relative durations `scheduler.interfaces.PatchSchedulingRequest`
and Section 14's own worked example already use ("a mission starting in
12 minutes, a patch requiring 4 minutes, and a communication window
closing in 6 minutes") -- CyberExposure(delay) is MTES(v) integrated over
a window of duration `t_window` starting at `now` (the vulnerability
remains exposed for the whole remaining window if the patch is delayed);
CyberExposure(patch_now) is MTES(v) integrated over the much shorter
window of duration `t_patch` (the vulnerability's residual exposure ends
once the patch is applied, `t_patch` time units from now).
"""

from __future__ import annotations

from dataclasses import dataclass

from cyber.metrics import MTESResult, mission_trust_exposure_score
from graph.trmg.model import TemporalReasoningMemoryGraph
from scheduler.interfaces import DELAY, PATCH_NOW


@dataclass(frozen=True)
class CyberExposureBreakdown:
    """The two MTES evaluations behind Algorithm 6, lines 1-2, kept
    alongside the plain `{action: value}` dict so a caller (and the
    Explainability Pipeline, Section 10.7) can trace each exposure value
    back to its own attenuation, decay, and criticality terms."""

    delay: MTESResult
    patch_now: MTESResult

    def as_dict(self) -> dict[str, float]:
        """The `PatchSchedulingRequest.cyber_exposure` shape Algorithm 6
        (scheduler/algorithm6.py) consumes."""
        return {PATCH_NOW: self.patch_now.value, DELAY: self.delay.value}


def compute_cyber_exposure(
    graph: TemporalReasoningMemoryGraph,
    vulnerability_node: str,
    mission_node: str,
    now: float,
    t_patch: float,
    t_window: float,
    *,
    intrinsic_risk: float,
    mission_criticality: float,
    k_hop: int = 6,
) -> CyberExposureBreakdown:
    """Algorithm 6, lines 1-2 (Section 14.1), computed from real MTES
    (Section 12.6.1, `cyber.metrics.mission_trust_exposure_score`) rather
    than a caller-supplied placeholder. See module docstring for why
    `t_patch`/`t_window` are treated as durations, not absolute times.
    """
    if t_patch < 0.0:
        raise ValueError(f"t_patch must be >= 0, got {t_patch}")
    if t_window < 0.0:
        raise ValueError(f"t_window must be >= 0, got {t_window}")

    delay = mission_trust_exposure_score(
        graph,
        vulnerability_node,
        mission_node,
        now,
        window=t_window,
        intrinsic_risk=intrinsic_risk,
        mission_criticality=mission_criticality,
        k_hop=k_hop,
    )
    patch_now = mission_trust_exposure_score(
        graph,
        vulnerability_node,
        mission_node,
        now,
        window=t_patch,
        intrinsic_risk=intrinsic_risk,
        mission_criticality=mission_criticality,
        k_hop=k_hop,
    )
    return CyberExposureBreakdown(delay=delay, patch_now=patch_now)
