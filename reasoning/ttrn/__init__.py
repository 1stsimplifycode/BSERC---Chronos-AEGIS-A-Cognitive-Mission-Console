"""TTRN: CMI's core reasoning engine (manuscript Section 5).

Restates the nine-stage TTRN pipeline (Table 1, Section 5.1) as
Algorithm 1 (Section 5.2), identified with the CMI state-transition
function sigma(m,t) = Phi(sigma(m,t-1), e_t) (Section 4.1).
"""

from reasoning.ttrn.pipeline import TelemetryEdge, TTRNPipeline, TTRNResult

__all__ = ["TelemetryEdge", "TTRNPipeline", "TTRNResult"]
