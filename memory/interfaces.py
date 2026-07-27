"""Interface Episodic Mission Memory (EMM, Section 8) must satisfy.

Algorithm 1 (Section 5.2) interfaces with EMM at exactly two points:
    line 3:  M_ret <- EMM-Retrieve(N_k(m))          (Algorithm 3, Section 8.4)
    line 15: EMM-Update(m, t, delta, T_post, E, C*)  (Algorithm 2, Section 8.3)

`EpisodicMemoryInterface` fixes that contract as a Protocol so
reasoning/ttrn/pipeline.py can depend on it without depending on a specific
concrete implementation. `memory.emm.EpisodicMissionMemory` is now the real
four-tier implementation (Section 8.2); `NullEpisodicMemory` below remains
available as an explicit no-persistence opt-out.

Both methods take an already-computed `FeatureVector` (memory/features.py)
rather than a raw TRMG neighborhood, so this module does not need to depend
on graph/trmg/model.py: the caller (the TTRN pipeline) extracts `s` once
per query, via the same extractor for both Algorithm 2 line 2 and Algorithm
3 line 1, and passes it to both calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.hypothesis import AgentHypothesis
from memory.features import FeatureVector
from memory.outcome import MissionOutcome


@dataclass(frozen=True)
class MemoryRetrievalResult:
    """M_ret(m,t): what Algorithm 3 (Section 8.4, line 7) returns --
    ranked episodic candidates together with any relevant semantic rules
    and meta-memory calibration entries."""

    episodes: tuple[Any, ...] = field(default_factory=tuple)
    semantic_rules: tuple[Any, ...] = field(default_factory=tuple)
    meta_calibration: dict[str, Any] = field(default_factory=dict)


class EpisodicMemoryInterface(Protocol):
    """The contract Section 8's Episodic Mission Memory must implement."""

    def retrieve(self, query: FeatureVector, now: float) -> MemoryRetrievalResult:
        """Algorithm 3 (Section 8.4): retrieve memory relevant to feature
        vector `query` (already extracted from N_k(m) by the caller)."""
        ...

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
        """Algorithm 2 (Section 8.3): write to working memory, and, if
        `outcome` is not None (Algorithm 2, line 3's `PendingOutcome(m)` --
        supplied by the caller here rather than looked up internally, since
        no ground-truth source such as the Mission Digital Twin, Section
        15, exists yet), append the resolved episode and become eligible
        for batched consolidation into semantic/meta memory."""
        ...


class NullEpisodicMemory:
    """An explicit no-persistence EpisodicMemoryInterface: use when a
    caller wants to run the TTRN pipeline without writing to any memory
    store at all (e.g. a stateless single-shot inference, or a test that
    does not care about memory).

    `retrieve` returns an empty MemoryRetrievalResult: "no episodic memory
    has been retrieved" is the true state of a system configured with no
    memory backend, not a fabricated result. `update` raises
    NotImplementedError -- silently discarding a write would misrepresent
    that the episode was actually persisted; callers
    (reasoning/ttrn/pipeline.py) catch this and record an explicit warning
    on the returned TTRNResult rather than letting the whole inference call
    fail. To persist episodes, pass `memory=EpisodicMissionMemory(...)`
    (memory/emm.py) to TTRNPipeline instead.
    """

    def retrieve(self, query: FeatureVector, now: float) -> MemoryRetrievalResult:
        return MemoryRetrievalResult()

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
        raise NotImplementedError(
            "No Episodic Mission Memory backend is configured; EMM-Update "
            "(Algorithm 2) cannot persist this episode. Pass "
            "memory=EpisodicMissionMemory(...) (memory/emm.py) to TTRNPipeline "
            "to enable persistence."
        )
