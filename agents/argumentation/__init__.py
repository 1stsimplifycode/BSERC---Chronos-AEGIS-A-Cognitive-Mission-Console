"""Structured AI Argumentation (manuscript Section 9.2-9.7).

`StructuredArgumentAggregator` (aggregation.py) is the real implementation
of Algorithm 4; `ArgumentationGraph` (graph.py) and the Dung-style
grounded/preferred extension semantics (semantics.py) are its building
blocks; `conflict_score`/`decide` (consensus.py) implement Section 9.3's
formulas. `NullArgumentationReconciler` (interfaces.py) is an explicit
no-argumentation opt-out.
"""

from agents.argumentation.aggregation import StructuredArgumentAggregator
from agents.argumentation.consensus import ConsensusResult, conflict_score, decide
from agents.argumentation.graph import ArgumentationGraph, build_argumentation_graph
from agents.argumentation.interfaces import (
    ArgumentationReconcilerInterface,
    NullArgumentationReconciler,
    ReconciledDecision,
)
from agents.argumentation.semantics import (
    grounded_extension,
    is_admissible,
    is_conflict_free,
    preferred_extensions,
)

__all__ = [
    "StructuredArgumentAggregator",
    "ConsensusResult",
    "conflict_score",
    "decide",
    "ArgumentationGraph",
    "build_argumentation_graph",
    "ArgumentationReconcilerInterface",
    "NullArgumentationReconciler",
    "ReconciledDecision",
    "grounded_extension",
    "is_admissible",
    "is_conflict_free",
    "preferred_extensions",
]
