"""Algorithm 4: Structured Argument Aggregation (manuscript Section 9.4).

Algorithm 4: ArgumentAggregate({H_1,...,H_9})
    1: Construct AG = (H, Attacks) from the agents' counter sets
    2: Retrieve credibility weights w_a for each agent a from Meta Memory M_m
    3: Compute Conflict(H) per Section 9.3
    4: Extension(AG) <- compute the grounded (or preferred, if the grounded
       extension is empty) extension of AG
    5: delta(m,t) <- argmax over claims of sum w*conf restricted to
       H in Extension(AG)  // Section 9.3
    6: if Extension(AG) is empty even under the preferred semantics then
    7:     Flag delta(m,t) as unresolved and escalate to a human operator,
           attaching Conflict(H) and AG  // no silent default
    8: end if
    9: return delta(m,t), AG, Conflict(H)
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.argumentation.consensus import conflict_score, decide
from agents.argumentation.graph import ArgumentationGraph, build_argumentation_graph
from agents.argumentation.interfaces import ReconciledDecision
from agents.argumentation.semantics import grounded_extension, preferred_extensions
from agents.hypothesis import AgentHypothesis
from memory.meta import MetaMemory


@dataclass(frozen=True)
class _CandidateExtension:
    agent_ids: frozenset[str]
    claim: str | None
    score: float


class StructuredArgumentAggregator:
    """The real ArgumentationReconcilerInterface implementation: Algorithm
    4 over Meta Memory M_m's credibility weights.

    `meta_memory` is dependency-injected rather than looked up globally, so
    a caller wires in the *same* MetaMemory instance their
    EpisodicMissionMemory uses (`memory.emm.EpisodicMissionMemory.meta`) if
    they want credibility to reflect real historical calibration; passing a
    fresh, empty `MetaMemory()` is equally valid and simply means every
    agent starts at `default_credibility`.
    """

    def __init__(self, meta_memory: MetaMemory, default_credibility: float = 0.5) -> None:
        """`default_credibility` is applied to any agent Meta Memory has no
        calibration record for yet (Section 9.6: credibility is "learned
        from historical outcomes"; the manuscript does not specify a value
        for an agent with no history). 0.5 is a documented, neutral prior
        -- neither trusting nor distrusting an unproven agent -- consistent
        with the uninformative-prior convention used elsewhere in this
        codebase (e.g. reasoning/trust/bayesian.py's mean=0.5 default).
        """
        if not 0.0 <= default_credibility <= 1.0:
            raise ValueError(f"default_credibility must be in [0,1], got {default_credibility}")
        self.meta_memory = meta_memory
        self.default_credibility = default_credibility

    def aggregate(self, hypotheses: tuple[AgentHypothesis, ...]) -> ReconciledDecision:
        if not hypotheses:
            return ReconciledDecision(
                claim=None, summary="no hypotheses were generated; no decision to reconcile"
            )

        # Line 1: construct AG.
        graph = build_argumentation_graph(hypotheses)

        # Line 2: retrieve credibility weights. Uses an explicit None check
        # (not `or`) so a legitimately zero credibility -- a maximally
        # miscalibrated agent -- is not mistaken for "no history yet" and
        # silently replaced with the neutral default.
        credibility = {}
        for agent_id in graph.agent_ids:
            known = self.meta_memory.credibility(agent_id)
            credibility[agent_id] = known if known is not None else self.default_credibility

        # Line 3: Conflict(H).
        conflict = conflict_score(graph, credibility)

        # Line 4: grounded extension, falling back to preferred.
        extension = grounded_extension(graph.agent_ids, graph.attacks)
        if not extension:
            candidates = preferred_extensions(graph.agent_ids, graph.attacks)
            extension = self._select_preferred(candidates, graph, credibility)

        # Lines 6-8: escalate on genuine non-resolution.
        if not extension:
            return ReconciledDecision(
                claim=None,
                argument_graph=graph,
                conflict_score=conflict,
                credibility_weights=credibility,
                escalated=True,
                summary=(
                    f"Structured AI Argumentation could not resolve {len(hypotheses)} "
                    f"hypotheses: no argument set is jointly acceptable even under the "
                    f"preferred semantics. Escalated to human operator (Algorithm 4, "
                    f"lines 6-8)."
                ),
            )

        # Line 5: consensus decision.
        result = decide(graph, extension, credibility)
        extension_hypotheses = tuple(
            sorted((h for h in hypotheses if h.agent_id in extension), key=lambda h: h.agent_id)
        )

        summary = (
            f"claim '{result.claim}' accepted from {len(extension)} of {len(hypotheses)} arguments"
        )
        return ReconciledDecision(
            claim=result.claim,
            supporting_hypotheses=result.supporting_hypotheses,
            extension_hypotheses=extension_hypotheses,
            argument_graph=graph,
            conflict_score=conflict,
            credibility_weights=credibility,
            escalated=False,
            summary=summary,
        )

    @staticmethod
    def _select_preferred(
        candidates: tuple[frozenset[str], ...],
        graph: ArgumentationGraph,
        credibility: dict[str, float],
    ) -> frozenset[str]:
        """When multiple preferred extensions exist, Section 9.3-9.4 do not
        specify which one to use. This picks the one whose eventual
        decision (Section 9.3's own argmax criterion) scores highest --
        i.e. it evaluates each candidate extension by the same
        credibility*confidence criterion the final decision is made with,
        rather than an unrelated, arbitrary tie-break. Ties among that are
        broken by candidate size (larger jointly-acceptable set preferred),
        then by sorted agent-id tuple for full determinism.
        """
        if not candidates:
            return frozenset()

        scored = []
        for candidate in candidates:
            result = decide(graph, candidate, credibility)
            score = sum(
                credibility[h.agent_id] * h.confidence for h in result.supporting_hypotheses
            )
            scored.append(_CandidateExtension(agent_ids=candidate, claim=result.claim, score=score))

        best = max(scored, key=lambda c: (c.score, len(c.agent_ids), sorted(c.agent_ids)))
        return best.agent_ids
