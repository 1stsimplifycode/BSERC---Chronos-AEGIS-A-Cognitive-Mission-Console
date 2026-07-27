"""Interface Structured AI Argumentation (Section 9) must satisfy.

Algorithm 1, line 10:

    delta(m,t) <- ArgumentAggregate({H_1, ..., H_9})   (Algorithm 4, Section 9.4)

`agents.argumentation.aggregation.StructuredArgumentAggregator` is the real
implementation of Algorithm 4 (Sections 9.2-9.4, 9.7); `NullArgumentationReconciler`
below remains as an explicit no-argumentation opt-out, analogous to
`memory.interfaces.NullEpisodicMemory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.hypothesis import AgentHypothesis


@dataclass(frozen=True)
class ReconciledDecision:
    """delta(m,t): the output of Structured Argument Aggregation (Algorithm
    4), carrying everything the Explainable Decision Engine (Section 9.7)
    requires to make the decision fully traceable: "the identity of every
    agent whose argument was accepted into Extension(AG), the specific
    TRMG nodes and memory entries each accepted argument cited as
    evidence, the credibility weight w applied to each, and the Conflict(H)
    score of the argument set as a whole."
    """

    claim: str | None = None
    """The winning claim, or None if unresolved/escalated or if no
    hypotheses were supplied at all."""

    supporting_hypotheses: tuple[AgentHypothesis, ...] = field(default_factory=tuple)
    """The subset of Extension(AG) whose claim equals `claim` -- the
    arguments that actually contributed to the winning score (Section 9.3's
    argmax)."""

    extension_hypotheses: tuple[AgentHypothesis, ...] = field(default_factory=tuple)
    """Every hypothesis accepted into Extension(AG), regardless of which
    claim it supports -- Section 9.7's full traceability requirement."""

    argument_graph: Any | None = None
    """AG = (H, Attacks) (Section 9.2); typed Any here to avoid this
    interface module depending on agents.argumentation.graph."""

    conflict_score: float | None = None
    """Conflict(H) (Section 9.3)."""

    credibility_weights: dict[str, float] = field(default_factory=dict)
    """w_a per agent, as retrieved from Meta Memory (Algorithm 4, line 2)."""

    escalated: bool = False
    """True iff Extension(AG) was empty even under the preferred semantics
    (Algorithm 4, lines 6-8): a genuine, non-silent escalation to a human
    operator, not a fabricated fallback decision."""

    summary: str = ""


class ArgumentationReconcilerInterface(Protocol):
    """The contract Section 9's Structured AI Argumentation must implement."""

    def aggregate(self, hypotheses: tuple[AgentHypothesis, ...]) -> ReconciledDecision:
        """ArgumentAggregate({H_1, ..., H_9}) (Algorithm 4, Section 9.4)."""
        ...


class NullArgumentationReconciler:
    """An explicit no-argumentation ArgumentationReconcilerInterface: use
    when a caller wants to run the TTRN pipeline without Structured AI
    Argumentation at all.

    Given zero hypotheses, "no decision to reconcile" is the true answer,
    not a fabrication. Given one or more hypotheses, raises
    NotImplementedError rather than approximate a consensus function this
    class does not implement. To get real argumentation, pass
    `argumentation=StructuredArgumentAggregator(meta_memory=...)`
    (agents/argumentation/aggregation.py) to TTRNPipeline instead.
    """

    def aggregate(self, hypotheses: tuple[AgentHypothesis, ...]) -> ReconciledDecision:
        if not hypotheses:
            return ReconciledDecision(
                claim=None, summary="no hypotheses were generated; no decision to reconcile"
            )
        raise NotImplementedError(
            "No Structured AI Argumentation backend is configured; cannot reconcile "
            "a non-empty hypothesis set. Pass argumentation=StructuredArgumentAggregator(...) "
            "(agents/argumentation/aggregation.py) to TTRNPipeline to enable reconciliation."
        )
