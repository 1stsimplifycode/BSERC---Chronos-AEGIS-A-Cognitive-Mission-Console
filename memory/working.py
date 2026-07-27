"""Working Memory M_w (manuscript Section 8.2.1).

"M_w holds the current mission state: the current TRMG slice N_k(m), the
current telemetry batch, current cyber events, the current set of agent
hypotheses {H_1,...,H_9}, current uncertainty Upsilon, and current trust
T_post. M_w is volatile and is overwritten on each query cycle of Algorithm
1, except for the subset selected for consolidation into M_e (Section
8.3)."

Cyber events are not modeled here: they belong to Sections 11-13
(`cyber/`), not yet implemented, so `WorkingMemorySnapshot` has no field
for them rather than a fabricated always-empty placeholder field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.hypothesis import AgentHypothesis


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    """The current query cycle's state, as held in M_w."""

    mission_id: str
    time: float
    reconciled_decision: Any
    trust_posterior: float
    evidence: Any
    causal_context: Any
    uncertainty: float | None = None
    agent_hypotheses: tuple[AgentHypothesis, ...] = field(default_factory=tuple)


class WorkingMemory:
    """M_w: volatile, single-slot, overwritten every query cycle (Section 8.2.1)."""

    def __init__(self) -> None:
        self._current: WorkingMemorySnapshot | None = None

    def write(self, snapshot: WorkingMemorySnapshot) -> None:
        """Algorithm 2, line 1: overwrite the previous query's entry."""
        self._current = snapshot

    @property
    def current(self) -> WorkingMemorySnapshot | None:
        return self._current
