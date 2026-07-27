"""Causal Mission Intelligence (manuscript Section 10).

reasoning.causal.scm             Structural Causal Model (Section 10.2)
reasoning.causal.counterfactual  The retained correlational Counterfactual
                                  Engine (Section 10.1)
reasoning.causal.algorithm5      Algorithm 5: Causal Counterfactual
                                  Generation (Section 10.5), the real
                                  CausalEngineInterface implementation
reasoning.causal.explainability  The eight-level Explainability Pipeline
                                  (Section 10.7)
reasoning.causal.interfaces      CausalEngineInterface contract +
                                  NullCausalEngine no-op opt-out

Scope note (Section 10.4): causal discovery -- inferring which TRMG edges
are causal from data -- is explicitly not solved by the manuscript and is
not implemented here; `StructuralCausalModel` is populated from
domain-supplied structural functions, not learned.
"""

from reasoning.causal.algorithm5 import (
    CandidateIntervention,
    CausalQueryResult,
    StructuralCausalEngine,
    parse_edge_variable,
)
from reasoning.causal.counterfactual import EdgeInterventionRequest, correlational_counterfactual
from reasoning.causal.explainability import (
    ExplanationInputs,
    ExplanationLevel,
    ExplanationReport,
    assemble_explanation,
)
from reasoning.causal.interfaces import CausalContext, CausalEngineInterface, NullCausalEngine
from reasoning.causal.scm import StructuralCausalModel

__all__ = [
    "CandidateIntervention",
    "CausalQueryResult",
    "StructuralCausalEngine",
    "parse_edge_variable",
    "EdgeInterventionRequest",
    "correlational_counterfactual",
    "ExplanationInputs",
    "ExplanationLevel",
    "ExplanationReport",
    "assemble_explanation",
    "CausalContext",
    "CausalEngineInterface",
    "NullCausalEngine",
    "StructuralCausalModel",
]
