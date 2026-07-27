"""Continual-Learning Convergence Benchmark (Phase 8; RQ10, Section 3:
"Does online continual learning (Section 16) reduce mission-planning
errors over successive missions?").

Runs a sequence of missions through Algorithm 1 (`TTRNPipeline.infer`) and
Algorithm 7 (`reasoning.continual.algorithm7.mission_adapt`) in order,
recording the real task loss and EWC penalty (Section 16.2) computed at
each adaptation step. This is a *measurement* tool, not a claim of
convergence: Section 16.6 is explicit that "we have not evaluated whether
the specific regularization strength lambda and replay-batch composition
prevent catastrophic forgetting in practice" -- this benchmark produces
exactly the trajectory Section 16.6/RQ10 says still needs evaluating,
without asserting in advance what it will show.

A fresh `MissionDigitalTwin` (and therefore a fresh TRMG) is used for
each mission in the sequence -- `MissionDigitalTwin.__init__` always
constructs its own graph (Section 15's design, unchanged by this phase) --
while the mission-aware GNN (`theta`), the shared `EpisodicMissionMemory`
(`M_e`, `M_m`), and the `ContinualLearningState` (`theta*`, `F`) persist
and accumulate across the whole sequence. This mirrors how Algorithm 7
would run in a deployed setting processing one mission's fresh telemetry
at a time while its learned parameters and memory persist between
missions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from digital_twin.integration import run_scenario_with_pipeline
from digital_twin.twin import MissionDigitalTwin
from evaluation.batch import ScenarioBuilder
from memory.emm import EpisodicMissionMemory
from reasoning.continual.algorithm7 import ContinualLearningState, initial_state, mission_adapt
from reasoning.trust.gnn import MissionAwareGNN
from reasoning.ttrn.pipeline import TTRNPipeline


@dataclass(frozen=True)
class ConvergencePoint:
    """One mission's own adaptation step in the sequence."""

    mission_index: int
    scenario_name: str
    task_loss: float
    ewc_penalty: float
    replay_batch_size: int


def continual_learning_convergence(
    scenario_builders: tuple[ScenarioBuilder, ...],
    seed: int,
    *,
    embedding_dim: int = 16,
    k_hop: int = 6,
    ewc_lambda: float = 0.1,
    learning_rate: float = 0.01,
    replay_batch_size: int = 8,
) -> tuple[ConvergencePoint, ...]:
    """Run every scenario in `scenario_builders`, in order, as one
    sequence of missions sharing a GNN/memory/continual-learning state,
    calling `mission_adapt` after each concludes. Returns one
    `ConvergencePoint` per mission, in sequence order -- the real
    trajectory of task loss (and EWC penalty) as adaptation proceeds.
    """
    rng = np.random.default_rng(seed)
    gnn = MissionAwareGNN(embedding_dim=embedding_dim, feature_dim=8, seed=seed)
    memory = EpisodicMissionMemory()
    state: ContinualLearningState = initial_state(gnn)

    points: list[ConvergencePoint] = []
    for index, builder in enumerate(scenario_builders):
        spec = builder()
        twin = MissionDigitalTwin(spec.name, seed=int(rng.integers(0, 2**31 - 1)))
        pipeline = TTRNPipeline(twin.graph, gnn, k_hop=k_hop, memory=memory)
        run_scenario_with_pipeline(twin, spec, pipeline)

        result = mission_adapt(
            twin.graph,
            gnn,
            memory,
            state,
            now=twin.env.time,
            ewc_lambda=ewc_lambda,
            learning_rate=learning_rate,
            replay_batch_size=replay_batch_size,
            rng=rng,
        )
        state = result.next_state
        points.append(
            ConvergencePoint(
                mission_index=index,
                scenario_name=spec.name,
                task_loss=result.task_loss,
                ewc_penalty=result.ewc_penalty,
                replay_batch_size=len(result.replay_batch),
            )
        )

    return tuple(points)
