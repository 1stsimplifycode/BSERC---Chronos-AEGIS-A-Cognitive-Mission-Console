"""The Cognitive Mission State (manuscript Section 4.1).

sigma(m,t) is defined as the single latent object every CMI component reads
and updates:

    sigma(m,t) = < h_m(t), T_post(m,t), Upsilon(m,t), M_ret(m,t),
                   AG(m,t), delta(m,t), C*(m,t) >

where h_m(t) is the graph-attention embedding (Section 7.1), T_post(m,t) and
Upsilon(m,t) are the Bayesian trust posterior and its uncertainty (Section
7.3), M_ret(m,t) is the memory retrieved for the current query (Section
8.4), AG(m,t) is the current argumentation graph (Section 9.2), delta(m,t)
is the reconciled decision (Section 9.7), and C*(m,t) is the active
counterfactual/causal context (Section 10).

Every CMI component reads a coordinate of sigma(m,t) or is a function that
maps (sigma(m,t-1), e_t) to a new coordinate of sigma(m,t) -- the single
state-transition function sigma(m,t) = Phi(sigma(m,t-1), e_t), identified
with Algorithm 1 (Section 5.2).

Note on notation: the source PDF's text extraction strips several Greek
glyphs (rendered as blank characters in the raw text). Where the manuscript
prose makes the intended symbol unambiguous (e.g. "T_post(m,t) and
Upsilon(m,t) are the Bayesian trust posterior and its uncertainty"), this
module uses the descriptive field name (`uncertainty`) rather than guessing
at a specific Greek letter, and records the mapping here for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class CognitiveMissionState:
    """sigma(m,t): an immutable snapshot of CMI's belief about mission m at
    time t. Immutable by design -- Phi always produces a *new* state rather
    than mutating the previous one, matching the paper's functional framing
    sigma(m,t) = Phi(sigma(m,t-1), e_t).
    """

    mission_id: str
    time: float

    # h_m(t): graph-attention embedding (Section 7.1). None until the GNN
    # stage of the pipeline has run.
    embedding: tuple[float, ...] | None = None

    # T_post(m,t): Bayesian trust posterior mean (Section 7.3).
    trust_posterior: float | None = None

    # Upsilon(m,t): posterior uncertainty (Section 7.3, Section 4.2).
    uncertainty: float | None = None

    # M_ret(m,t): memory retrieved for this query (Section 8.4).
    # NOT YET IMPLEMENTED (Section 8) -- always empty in this phase.
    retrieved_memory: tuple[Any, ...] = field(default_factory=tuple)

    # AG(m,t): the current argumentation graph (Section 9.2).
    # NOT YET IMPLEMENTED (Section 9) -- always None in this phase.
    argument_graph: Any | None = None

    # delta(m,t): the reconciled decision (Section 9.7).
    # NOT YET IMPLEMENTED (Section 9) -- always None in this phase.
    reconciled_decision: Any | None = None

    # C*(m,t): the active counterfactual/causal context (Section 10).
    # NOT YET IMPLEMENTED (Section 10) -- always None in this phase.
    causal_context: Any | None = None

    def with_updates(self, **changes: Any) -> CognitiveMissionState:
        """Return a new state with the given coordinates replaced, leaving
        every other coordinate untouched. This is the mechanical form of
        Phi: each pipeline stage calls this once to write exactly the
        coordinate(s) it owns (Section 4.1)."""
        return replace(self, **changes)
