"""AI Contribution 2: Mission-Aware Graph Neural Network, Trust Propagation,
Bayesian Trust Propagation, and the Mission Criticality Model
(manuscript Section 7)."""

from reasoning.trust.bayesian import BayesianTrustPosterior, bayesian_trust_update
from reasoning.trust.criticality import MissionCriticalityModel
from reasoning.trust.gnn import MissionAwareGNN, PredictionHeadOutputs
from reasoning.trust.propagation import propagated_trust_contribution, total_propagated_trust_input

__all__ = [
    "BayesianTrustPosterior",
    "bayesian_trust_update",
    "MissionCriticalityModel",
    "MissionAwareGNN",
    "PredictionHeadOutputs",
    "propagated_trust_contribution",
    "total_propagated_trust_input",
]
