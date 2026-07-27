"""Interface Causal Mission Intelligence (Section 10) must satisfy.

Algorithm 1, line 14:

    E(m,t) <- AssembleEvidence(N_k(m), H_1..9, delta(m,t))
    C* <- CausalCounterfactual(delta(m,t), candidate interventions)  (Algorithm 5, Section 10.5)

`reasoning.causal.algorithm5.StructuralCausalEngine` is the real
implementation of Algorithm 5 (Sections 10.1-10.5); `NullCausalEngine`
below remains as an explicit no-causal-engine opt-out, analogous to
`memory.interfaces.NullEpisodicMemory` and
`agents.argumentation.interfaces.NullArgumentationReconciler`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CausalContext:
    """C*(m,t): the active counterfactual/causal context (Section 10).

    `results` holds one entry per evaluated candidate intervention
    (typed `Any` here to avoid this interface module depending on
    `reasoning.causal.algorithm5.CausalQueryResult` -- each entry carries
    the intervention, the resulting delta, and its causal-status label;
    see algorithm5.py).
    """

    results: tuple[Any, ...] = field(default_factory=tuple)
    summary: str = ""


class CausalEngineInterface(Protocol):
    """The contract Section 10's Causal Mission Intelligence must implement."""

    def generate_counterfactual(
        self, reconciled_decision: Any, candidate_interventions: tuple[Any, ...]
    ) -> CausalContext:
        """CausalCounterfactual(delta(m,t), candidate interventions) (Algorithm 5)."""
        ...


class NullCausalEngine:
    """An explicit no-causal-engine CausalEngineInterface: use when a
    caller wants to run the TTRN pipeline without Causal Mission
    Intelligence at all.

    Given zero candidate interventions, "no causal context to evaluate" is
    the true answer, not a fabrication. Given one or more candidates,
    raises NotImplementedError rather than fabricate a causal judgment. To
    get real causal counterfactuals, pass
    `causal_engine=StructuralCausalEngine(...)`
    (reasoning/causal/algorithm5.py) to TTRNPipeline instead.
    """

    def generate_counterfactual(
        self, reconciled_decision: Any, candidate_interventions: tuple[Any, ...]
    ) -> CausalContext:
        if not candidate_interventions:
            return CausalContext(summary="no candidate interventions were supplied")
        raise NotImplementedError(
            "No Causal Mission Intelligence backend is configured; cannot evaluate "
            "candidate interventions. Pass causal_engine=StructuralCausalEngine(...) "
            "(reasoning/causal/algorithm5.py) to TTRNPipeline to enable causal counterfactuals."
        )
