"""Algorithm 5: Causal Counterfactual Generation (manuscript Section 10.5).

Algorithm 5: CausalCounterfactual(delta(m,t), candidate interventions)
    1: for each candidate intervention do(X->x) in the candidate set do
    2:     if X (TRMG dependency edge) is marked causal in M (Section 10.2) then
    3:         Apply do(X->x) to M: replace f_X with the constant x, leaving
               other structural functions unchanged
    4:         Recompute descendants of X under F to obtain T | do(X->x)
    5:         DeltaT <- T|do(X->x) - T|x   // causal effect estimate
    6:     else
    7:         Recompute via the correlational GNN forward pass as in
               Section 10.1, and label the result explicitly as
               "correlational, not yet verified causal"
    8:     end if
    9: end for
    10: return ranked list of (intervention, T, causal-status label)

"Line 7's explicit labeling is deliberate: rather than presenting every
counterfactual with equal apparent certainty, Algorithm 5 distinguishes
interventions computed over an edge already marked causal (Section 10.2)
from those computed by re-running the correlational predictive model
alone, so that an operator is not misled into treating a correlational
counterfactual as a verified causal claim."

`CandidateIntervention.variable` names X. The causal branch (line 2) fires
when `variable` is a registered StructuralCausalModel variable (Section
10.2: "restricted to those [edges] for which a structural function f_v has
been specified"). The correlational branch requires `variable` to identify
a TRMG edge; by convention it must be of the form "source->target" (see
`parse_edge_variable`), since the GNN-based correlational engine (Section
10.1, reasoning/causal/counterfactual.py) only knows how to intervene on
edge confidences. A `variable` that is neither a registered SCM variable
nor a parseable, existing TRMG edge is a configuration error, not silently
skipped or fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from reasoning.causal.counterfactual import EdgeInterventionRequest, correlational_counterfactual
from reasoning.causal.interfaces import CausalContext
from reasoning.causal.scm import StructuralCausalModel
from reasoning.trust.gnn import MissionAwareGNN

CAUSAL = "causal"
CORRELATIONAL = "correlational, not yet verified causal"


@dataclass(frozen=True)
class CandidateIntervention:
    """do(X -> x): `variable` is X, `value` is x."""

    variable: str
    value: float


@dataclass(frozen=True)
class CausalQueryResult:
    """One row of Algorithm 5's line 10 return value."""

    intervention: CandidateIntervention
    delta_trust: float
    causal_status: str


def parse_edge_variable(variable: str) -> tuple[str, str]:
    """Parse the "source->target" convention the correlational branch uses
    to address a TRMG edge by a single `variable` string."""
    if "->" not in variable:
        raise ValueError(
            f"variable {variable!r} is not a registered causal variable and is not a "
            f"'source->target' TRMG edge reference; cannot evaluate it correlationally either"
        )
    source, _, target = variable.partition("->")
    return source, target


class StructuralCausalEngine:
    """The real CausalEngineInterface implementation: Algorithm 5 over a
    StructuralCausalModel (causal branch) and the retained correlational
    Counterfactual Engine (Section 10.1, fallback branch).
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        graph: TemporalReasoningMemoryGraph,
        gnn: MissionAwareGNN,
        mission_node: str,
        node_features: dict[str, np.ndarray],
        now: float,
        trust_variable: str = "mission_trust",
    ) -> None:
        self.scm = scm
        self.graph = graph
        self.gnn = gnn
        self.mission_node = mission_node
        self.node_features = node_features
        self.now = now
        self.trust_variable = trust_variable

    def generate_counterfactual(
        self, reconciled_decision: Any, candidate_interventions: tuple[CandidateIntervention, ...]
    ) -> CausalContext:
        if not candidate_interventions:
            return CausalContext(summary="no candidate interventions were supplied")

        results = tuple(self._evaluate_one(c) for c in candidate_interventions)
        ranked = tuple(sorted(results, key=lambda r: abs(r.delta_trust), reverse=True))

        n_causal = sum(1 for r in ranked if r.causal_status == CAUSAL)
        summary = (
            f"evaluated {len(ranked)} candidate intervention(s): {n_causal} causal, "
            f"{len(ranked) - n_causal} correlational-only (Algorithm 5, Section 10.5)"
        )
        return CausalContext(results=ranked, summary=summary)

    def _evaluate_one(self, candidate: CandidateIntervention) -> CausalQueryResult:
        # Line 2: is X marked causal in M?
        if candidate.variable in self.scm.variables:
            baseline = self.scm.evaluate()
            intervened = self.scm.do(candidate.variable, candidate.value)
            if self.trust_variable not in baseline or self.trust_variable not in intervened:
                raise KeyError(
                    f"trust_variable {self.trust_variable!r} is not a registered SCM variable; "
                    f"cannot compute a causal effect estimate on it"
                )
            delta = intervened[self.trust_variable] - baseline[self.trust_variable]
            return CausalQueryResult(
                intervention=candidate, delta_trust=delta, causal_status=CAUSAL
            )

        # Lines 6-7: correlational fallback via the retained GNN-based engine.
        source, target = parse_edge_variable(candidate.variable)
        delta = correlational_counterfactual(
            self.graph,
            self.gnn,
            self.mission_node,
            self.node_features,
            self.now,
            EdgeInterventionRequest(source=source, target=target, value=candidate.value),
        )
        return CausalQueryResult(
            intervention=candidate, delta_trust=delta, causal_status=CORRELATIONAL
        )
