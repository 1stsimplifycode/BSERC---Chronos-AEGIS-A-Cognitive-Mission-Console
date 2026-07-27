"""The Argumentation Graph (manuscript Section 9.2).

"The argumentation graph is then AG = (H, Attacks), where H = {H_1,...,H_9}
is the set of arguments and Attacks subset H x H is the relation defined by
counter. This is a direct extension of Dung-style abstract argumentation,
in which arguments and an attack relation determine which arguments are
jointly acceptable under a chosen semantics; we extend it with the
quantified conf and, below, an explicit agent credibility weight, neither
of which classical abstract argumentation requires."
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.hypothesis import AgentHypothesis


@dataclass(frozen=True)
class ArgumentationGraph:
    """AG = (H, Attacks) (Section 9.2). `attacks` holds
    (attacker_agent_id, attacked_agent_id) pairs."""

    hypotheses: tuple[AgentHypothesis, ...]
    attacks: frozenset[tuple[str, str]]

    @property
    def agent_ids(self) -> frozenset[str]:
        return frozenset(h.agent_id for h in self.hypotheses)

    def attackers_of(self, agent_id: str) -> frozenset[str]:
        return frozenset(a for a, b in self.attacks if b == agent_id)

    def hypothesis_for(self, agent_id: str) -> AgentHypothesis:
        for h in self.hypotheses:
            if h.agent_id == agent_id:
                return h
        raise KeyError(agent_id)


def build_argumentation_graph(hypotheses: tuple[AgentHypothesis, ...]) -> ArgumentationGraph:
    """Construct AG = (H, Attacks) from the agents' counter sets (Algorithm
    4, line 1).

    An attack referencing an `agent_id` not present in `hypotheses` (e.g. an
    agent's hypothesis was generated in a prior round, or the referenced
    agent abstained this round) is dropped rather than raising: AG must be
    constructed purely from *this* round's argument set H, and a dangling
    reference to an absent agent cannot attack anything within it. This is
    a deliberate leniency, not a silent bug -- `graph.attacks` only ever
    contains pairs where both endpoints are in `graph.agent_ids`.
    """
    agent_ids = {h.agent_id for h in hypotheses}
    attacks = frozenset(
        (h.agent_id, target)
        for h in hypotheses
        for target in h.counterarguments
        if target in agent_ids
    )
    return ArgumentationGraph(hypotheses=hypotheses, attacks=attacks)
