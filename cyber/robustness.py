"""Section 12.5: Adversarial AI Robustness.

"We identify four adversarial threat surfaces specific to an AI reasoning
framework operating over a shared graph and language-capable explanation
layer, and specify the reasoning-level (not merely implementation-level)
mitigation each requires":

1. Prompt injection resilience -- `has_evidence_backing` /
   `require_evidence_backing` below.
2. Model poisoning resilience -- `is_corroborated` below.
3. Sensor spoofing resilience -- NOT a new mechanism here. The manuscript
   specifies this is handled "via cross-agent argumentation (Section 9),
   since a spoofed sensor reading that conflicts with another agent's
   independent evidence is exactly the kind of disagreement Structured AI
   Argumentation is designed to surface rather than silently average
   away." `agents.argumentation` (Section 9, Phase 3) already builds an
   Attacks relation from conflicting hypotheses and never silently
   averages a conflict away (Algorithm 4 either resolves via the
   grounded/preferred extension or escalates to a human operator) -- this
   module does not duplicate that mechanism. The other half of Section
   12.5's sensor-spoofing mitigation, "the existing sensor-spoofing
   scenario in the simulation library (Section 15.1)," requires the
   Mission Digital Twin (Section 15), which is explicitly out of scope
   for this phase.
4. Communication integrity reasoning -- `communication_link_trust` below.
"""

from __future__ import annotations

from agents.hypothesis import AgentHypothesis
from cyber.risk_propagation import dependency_node_trust
from graph.trmg.model import TemporalReasoningMemoryGraph

# -- 1. Prompt injection resilience ------------------------------------------


def has_evidence_backing(evidence: tuple[str, ...], graph: TemporalReasoningMemoryGraph) -> bool:
    """ "[A]ny such [natural-language evidence] summary is treated as
    untrusted content rather than as an instruction, and the
    reconciliation and scheduling layers (Sections 9, 14) never execute an
    action based on text content alone without a corresponding TRMG
    evidence node backing it" (Section 12.5).

    True only if `evidence` is non-empty and every cited identifier
    corresponds to an actual node in the TRMG -- a claim backed by no
    evidence, or by identifiers that do not resolve to real graph nodes,
    cannot pass regardless of how persuasive its text is.
    """
    if not evidence:
        return False
    return all(graph.has_node(e) for e in evidence)


def require_evidence_backing(
    hypothesis: AgentHypothesis, graph: TemporalReasoningMemoryGraph
) -> None:
    """Raises ValueError if `hypothesis.evidence` is not TRMG-backed
    (Section 12.5). Intended as a guard at the reconciliation/scheduling
    boundary (Sections 9, 14) the manuscript names explicitly.
    """
    if not has_evidence_backing(hypothesis.evidence, graph):
        raise ValueError(
            f"hypothesis from agent {hypothesis.agent_id!r} (claim={hypothesis.claim!r}) is not "
            f"backed by any TRMG evidence node; refusing to treat its text content as actionable "
            f"(Section 12.5, prompt injection resilience)"
        )


# -- 2. Model poisoning resilience -------------------------------------------


def is_corroborated(primary_source: str, corroborating_sources: tuple[str, ...]) -> bool:
    """ "[O]utcome records feeding continual learning must be corroborated
    by an independent telemetry source before being accepted into
    Episodic Memory (Section 8.3), rather than accepted from a single
    reporting channel" (Section 12.5).

    True iff at least one source in `corroborating_sources` is independent
    of (i.e. not equal to) `primary_source`.

    NOTE: Section 16 (the online continual-learning loop) is the mechanism
    that would call this at EMM-Update time and is explicitly out of scope
    for this phase (see docs/ARCHITECTURE.md); this is the real, tested
    reasoning-level check Section 16 will need, provided now rather than
    left for a future implementation to invent.
    """
    independent = set(corroborating_sources) - {primary_source}
    return len(independent) >= 1


# -- 4. Communication integrity reasoning ------------------------------------


def communication_link_trust(
    graph: TemporalReasoningMemoryGraph, link_node: str, now: float, decay_rate: float = 0.05
) -> float:
    """ "[C]ommunication-link trust is treated as a TRMG node subject to the
    same confidence-decay mechanism as any other evidence" (Section 12.5).
    A thin semantic alias over `cyber.risk_propagation.dependency_node_trust`
    -- comm-link trust decays via exactly the same mechanism dependency
    trust does (Section 12.3), so this reuses that function rather than
    reimplementing decay for a third time.
    """
    node = graph.get_node(link_node)
    return dependency_node_trust(node, now, decay_rate)
