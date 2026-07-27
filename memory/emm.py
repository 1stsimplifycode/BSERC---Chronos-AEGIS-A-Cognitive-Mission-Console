"""Episodic Mission Memory (manuscript Section 8): EMM = (M_w, M_s, M_e, M_m).

`EpisodicMissionMemory` composes the four tiers (memory/working.py,
memory/semantic.py, memory/episodic.py, memory/meta.py) and implements
`EpisodicMemoryInterface` (memory/interfaces.py) as Algorithm 2 (Section
8.3, EMM-Update) and Algorithm 3 (Section 8.4, EMM-Retrieve), including the
batched consolidation of Section 8.3.1.
"""

from __future__ import annotations

from typing import Any

from agents.hypothesis import AgentHypothesis
from memory.consolidation import (
    ConsolidationConfig,
    ConsolidationReport,
    ConsolidationSchedule,
    consolidate,
)
from memory.episodic import EpisodicEntry, EpisodicMemoryStore
from memory.features import FeatureVector
from memory.interfaces import MemoryRetrievalResult
from memory.meta import MetaMemory
from memory.outcome import MissionOutcome
from memory.semantic import SemanticMemory
from memory.similarity import SimilarityWeights
from memory.working import WorkingMemory, WorkingMemorySnapshot


class EpisodicMissionMemory:
    """EMM = (M_w, M_s, M_e, M_m) (Section 8.2), implementing
    EpisodicMemoryInterface."""

    def __init__(
        self,
        *,
        consolidation_config: ConsolidationConfig | None = None,
        top_k: int = 5,
        similarity_weights: SimilarityWeights | None = None,
        recency_decay_rate: float = 0.01,
    ) -> None:
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemoryStore()
        self.meta = MetaMemory()
        self.consolidation_config = consolidation_config or ConsolidationConfig()
        self._schedule = ConsolidationSchedule(self.consolidation_config)
        self.top_k = top_k
        self.similarity_weights = similarity_weights
        self.recency_decay_rate = recency_decay_rate
        self._last_consolidation: ConsolidationReport | None = None

    @property
    def last_consolidation(self) -> ConsolidationReport | None:
        """The most recent batched-consolidation report, or None if
        consolidation has not run yet (Section 8.3.1)."""
        return self._last_consolidation

    def retrieve(self, query: FeatureVector, now: float) -> MemoryRetrievalResult:
        """Algorithm 3 (Section 8.4): EMM-Retrieve(N_k(m)), given the
        feature vector already extracted from N_k(m)."""
        candidates = self.episodic.search(
            query, self.top_k, now, self.similarity_weights, self.recency_decay_rate
        )
        rule = self.semantic.query(query.dependency_identity, query.topology_class)
        semantic_rules = (rule,) if rule is not None else ()

        agent_ids = {
            hypothesis.agent_id for c in candidates for hypothesis in c.entry.agent_hypotheses
        }
        meta_calibration = {
            agent_id: calib
            for agent_id, calib in self.meta.all_calibrations().items()
            if agent_id in agent_ids
        }

        return MemoryRetrievalResult(
            episodes=candidates, semantic_rules=semantic_rules, meta_calibration=meta_calibration
        )

    def update(
        self,
        mission_id: str,
        time: float,
        reconciled_decision: Any,
        trust_posterior: float,
        evidence: Any,
        causal_context: Any,
        features: FeatureVector,
        agent_hypotheses: tuple[AgentHypothesis, ...] = (),
        outcome: MissionOutcome | None = None,
        outcome_confidence: float = 1.0,
        embedding: tuple[float, ...] = (),
    ) -> None:
        """Algorithm 2 (Section 8.3): EMM-Update(m, t, delta, T, E, C*).

        Phase 8 (Section 16.1's "mission outcome -> memory update" step):
        when `outcome` is not None and `evidence` is the
        `MemoryRetrievalResult` this same mission's own EMM-Retrieve call
        returned, every retrieved episode's retrieval confidence is
        reinforced (if its own outcome agrees with the now-realized
        `outcome`) or weakened (if it disagrees) via the non-destructive
        `EpisodicMemoryStore.reliability` ledger -- see
        `memory/reliability.py`. Retrieved episodes are never deleted or
        mutated, only their *effective* future ranking is adjusted.
        """
        # Line 1: write working memory, overwriting the previous entry.
        self.working.write(
            WorkingMemorySnapshot(
                mission_id=mission_id,
                time=time,
                reconciled_decision=reconciled_decision,
                trust_posterior=trust_posterior,
                evidence=evidence,
                causal_context=causal_context,
                agent_hypotheses=agent_hypotheses,
            )
        )

        # Lines 3-5: PendingOutcome(m) is supplied by the caller as `outcome`
        # (see memory/interfaces.py docstring for why); if not None, append.
        if outcome is not None:
            if isinstance(evidence, MemoryRetrievalResult):
                for candidate in evidence.episodes:
                    self.episodic.reliability.reinforce_from_outcome(
                        candidate.entry,
                        outcome,
                        time=time,
                        source_mission_id=mission_id,
                    )

            entry = EpisodicEntry(
                mission_id=mission_id,
                time=time,
                features=features,
                reconciled_decision=reconciled_decision,
                trust=trust_posterior,
                agent_hypotheses=agent_hypotheses,
                causal_context=causal_context,
                outcome=outcome,
                outcome_confidence=outcome_confidence,
                embedding=embedding,
            )
            self.episodic.append(entry)
            self._schedule.record_completed_mission()

        # Lines 6-9, 11 (Section 8.3.1): batched, not on every call.
        if self._schedule.due(len(self.episodic)):
            self._last_consolidation = consolidate(
                self.episodic, self.semantic, self.meta, self.consolidation_config
            )
            self._schedule.reset()
