"""Episodic Mission Memory (EMM) -- manuscript Section 8.

EMM = (M_w, M_s, M_e, M_m): four memory tiers -- working (memory/working.py),
semantic (memory/semantic.py), episodic (memory/episodic.py), and meta
(memory/meta.py) -- composed by `EpisodicMissionMemory` (memory/emm.py),
which implements Algorithm 2 (Memory Update and Consolidation, Section 8.3)
and Algorithm 3 (Memory Retrieval, Section 8.4).

`EpisodicMemoryInterface` (memory/interfaces.py) is the contract
reasoning/ttrn/pipeline.py depends on; `NullEpisodicMemory` is an explicit
no-persistence opt-out for callers who don't want a memory backend.
"""

from memory.emm import EpisodicMissionMemory
from memory.episodic import EpisodicEntry, EpisodicMemoryStore, RankedEpisodicCandidate
from memory.features import FeatureVector, extract_feature_vector
from memory.interfaces import EpisodicMemoryInterface, MemoryRetrievalResult, NullEpisodicMemory
from memory.meta import AgentCalibration, MetaMemory
from memory.outcome import MissionOutcome
from memory.semantic import SemanticMemory, SemanticRule
from memory.similarity import SimilarityWeights
from memory.working import WorkingMemory, WorkingMemorySnapshot

__all__ = [
    "EpisodicMissionMemory",
    "EpisodicEntry",
    "EpisodicMemoryStore",
    "RankedEpisodicCandidate",
    "FeatureVector",
    "extract_feature_vector",
    "EpisodicMemoryInterface",
    "MemoryRetrievalResult",
    "NullEpisodicMemory",
    "AgentCalibration",
    "MetaMemory",
    "MissionOutcome",
    "SemanticMemory",
    "SemanticRule",
    "SimilarityWeights",
    "WorkingMemory",
    "WorkingMemorySnapshot",
]
