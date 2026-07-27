"""Sections 11-13: Cognitive Cyber Reasoning, Mission-Aware Cyber Risk
Propagation (base mechanism, attack-chain reasoning, supply-chain/firmware
trust, zero-trust reasoning, adversarial AI robustness, mission-aware
cyber metrics), and the Temporal Attack Graph.

cyber.consequence           Shared mu(q) similarity-weighted mission-
                             consequence formula (Sections 11, 12.2)
cyber.cognitive_reasoning   Section 11: Cognitive Cyber Reasoning
cyber.risk_propagation      Section 12.1 (base propagation), 12.3
                             (dependency/firmware trust), 12.4 (zero-trust)
cyber.attack_chain          Section 12.2: AI-Driven Attack Chain Reasoning
cyber.robustness            Section 12.5: Adversarial AI Robustness
cyber.centrality            Betweenness centrality over the dependency
                             subgraph, for Section 12.6.2 (MDRI)
cyber.metrics               Section 12.6: MTES, MDRI, CMCS
cyber.attack_graph          Section 13: Temporal Attack Graph
cyber.scheduler_integration Bridges MTES into Algorithm 6's CyberExposure
                             (scheduler/algorithm6.py, Section 14.1 lines 1-2)

Every equation implemented here traces to a specific manuscript
subsection cited in the corresponding module's docstring; every place
this repository had to derive a detail the manuscript leaves
underspecified is flagged "ASSUMPTION (documented)" at the point it is
made, per the project's no-fabrication standard. See cyber/README.md for
the consolidated list and cross-references to the scheduler and
Explainability Pipeline integrations that consume this package's outputs.
"""

from cyber.attack_chain import (
    AttackChainReasoning,
    AttackChainRiskAdjustment,
    is_attack_chain_episode,
)
from cyber.attack_graph import (
    AttackPath,
    AttackPathScore,
    AttackStep,
    attack_path_feature_vector,
    mission_phase_imminence,
    path_structural_score,
    score_attack_path,
    validate_attack_path,
)
from cyber.centrality import (
    betweenness_centrality,
    dependency_nodes,
    dependency_subgraph_adjacency,
)
from cyber.cognitive_reasoning import (
    CognitiveCyberReasoning,
    CognitiveCyberReasoningResult,
    is_cyber_relevant_episode,
)
from cyber.consequence import ConsequencePrediction, predict_mission_consequence
from cyber.metrics import (
    CMCSResult,
    MDRIResult,
    MTESResult,
    cyber_mission_criticality_score,
    mission_dependency_risk_index,
    mission_trust_exposure_score,
)
from cyber.risk_propagation import (
    REFRESH_EVENTS,
    attenuation_to,
    dependency_node_trust,
    propagate_risk_attenuation,
    propagate_risk_attenuation_with_decay,
    refresh_dependency_trust,
    scoped_neighborhood_for_agent,
    verify_identity_trust,
)
from cyber.robustness import (
    communication_link_trust,
    has_evidence_backing,
    is_corroborated,
    require_evidence_backing,
)
from cyber.scheduler_integration import CyberExposureBreakdown, compute_cyber_exposure

__all__ = [
    "AttackChainReasoning",
    "AttackChainRiskAdjustment",
    "is_attack_chain_episode",
    "AttackPath",
    "AttackPathScore",
    "AttackStep",
    "attack_path_feature_vector",
    "mission_phase_imminence",
    "path_structural_score",
    "score_attack_path",
    "validate_attack_path",
    "betweenness_centrality",
    "dependency_nodes",
    "dependency_subgraph_adjacency",
    "CognitiveCyberReasoning",
    "CognitiveCyberReasoningResult",
    "is_cyber_relevant_episode",
    "ConsequencePrediction",
    "predict_mission_consequence",
    "CMCSResult",
    "MDRIResult",
    "MTESResult",
    "cyber_mission_criticality_score",
    "mission_dependency_risk_index",
    "mission_trust_exposure_score",
    "REFRESH_EVENTS",
    "attenuation_to",
    "dependency_node_trust",
    "propagate_risk_attenuation",
    "propagate_risk_attenuation_with_decay",
    "refresh_dependency_trust",
    "scoped_neighborhood_for_agent",
    "verify_identity_trust",
    "communication_link_trust",
    "has_evidence_backing",
    "is_corroborated",
    "require_evidence_backing",
    "CyberExposureBreakdown",
    "compute_cyber_exposure",
]
