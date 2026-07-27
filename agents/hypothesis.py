"""Multi-Agent Hypothesis Generation (manuscript Section 9.1, retained).

Every agent's hypothesis is a tuple of claim, evidence, confidence, and
counterarguments (Section 9). Algorithm 1, lines 7-9:

    for each agent a in A do
        H_a <- GenerateHypothesis(a, N_k(m), T_post, M_ret, domain-specific evidence)
    end for

Section 5.3 fixes |A| = 9 in the complexity analysis (the manuscript's
domain-specialized agent roster -- e.g. weather, maintenance, communications,
cyber -- is not enumerated in the sections read for this phase). This module
defines the hypothesis data structure and the generator interface only; it
does NOT provide the nine concrete domain agents, which is future work.
Lines 7-9 note agents are independent and executable in parallel; nothing in
this interface prevents that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentHypothesis:
    """H_a = (claim, evidence, conf, counter) (Section 9.2).

    `counterarguments` operationalizes Section 9.2's "counter is the set of
    other agents' claims that claim attacks (i.e., is inconsistent with)"
    as the set of `agent_id`s this hypothesis attacks -- agent_id is a
    stable identifier each round (one hypothesis per agent), whereas
    matching on raw claim text would be fragile. `agents/argumentation/graph.py`
    builds the Attacks relation directly from this field.
    """

    agent_id: str
    claim: str
    evidence: tuple[str, ...]
    confidence: float
    counterarguments: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.agent_id in self.counterarguments:
            raise ValueError(f"agent '{self.agent_id}' cannot attack its own hypothesis")


class HypothesisGeneratorInterface(Protocol):
    """The contract each of the |A|=9 domain-specialized agents implements."""

    agent_id: str

    def generate(
        self,
        neighborhood_node_ids: set[str],
        trust_posterior: float,
        retrieved_memory: Any,
        domain_evidence: Any,
    ) -> AgentHypothesis:
        """GenerateHypothesis(a, N_k(m), T_post, M_ret, domain-specific evidence)."""
        ...
