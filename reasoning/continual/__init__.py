"""Section 16: AI Contribution 7, Adaptive Learning and Continual Reasoning.

reasoning.continual.objective    Section 16.2's Regularized Continual
                                  Objective (task loss + EWC penalty) and
                                  Section 16.3's experience-replay item type
reasoning.continual.algorithm7   Algorithm 7 (Section 16.5): MissionAdapt

Section 16.4 (meta-learning) is explicitly a "design possibility, not a
claim" in the manuscript and is not implemented here -- see that
section's own text.
"""

from reasoning.continual.algorithm7 import (
    ContinualLearningState,
    MissionAdaptResult,
    initial_state,
    mission_adapt,
)
from reasoning.continual.objective import (
    HEAD_NAMES,
    ReplayItem,
    batch_task_loss_and_gradients,
    estimate_fisher_diagonal,
    ewc_penalty_and_gradients,
    outcome_targets,
    regularized_update,
    task_loss_and_gradients,
)

__all__ = [
    "ContinualLearningState",
    "MissionAdaptResult",
    "initial_state",
    "mission_adapt",
    "HEAD_NAMES",
    "ReplayItem",
    "batch_task_loss_and_gradients",
    "estimate_fisher_diagonal",
    "ewc_penalty_and_gradients",
    "outcome_targets",
    "regularized_update",
    "task_loss_and_gradients",
]
