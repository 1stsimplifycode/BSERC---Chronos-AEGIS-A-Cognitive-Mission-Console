"""Algorithm 1: TTRN-Infer(G_t, m, A) (manuscript Section 5.2).

    1: Apply decay lambda to all edge confidences in G_t since the last
       update; ingest new telemetry edges
    2: N_k(m) <- k-hop neighborhood of m in G_t (k <= 6)
    3: M_ret <- EMM-Retrieve(N_k(m))                          [Section 8.4]
    4: h_m(t) <- TemporalGraphAttention(N_k(m) union M_ret)   [Section 7.1]
    5: (T, P_success, D, C, R_comm, O) <- PredictionHeads(h_m(t))
    6: (T_post, Upsilon) <- BayesianUpdate(T, prior_{t-1})    [Section 7.3]
    7: for each agent a in A do
    8:     H_a <- GenerateHypothesis(a, N_k(m), T_post, M_ret, domain evidence)
    9: end for
    10: delta(m,t) <- ArgumentAggregate({H_1..9})             [Section 9.4]
    11: if the reconciled decision requires patch scheduling then
    12:     delta(m,t) <- AIPatchScheduler(delta(m,t), ...)   [Section 14]
    13: end if
    14: E(m,t) <- AssembleEvidence(...); C* <- CausalCounterfactual(...) [10.5]
    15: EMM-Update(m, t, delta(m,t), T_post, E(m,t), C*)      [Section 8.3]
    16: return delta(m,t), T_post, (E(m,t), C*)

This module implements every line of Algorithm 1 as real control flow.
Lines 3 and 15 depend on Section 8 (Episodic Mission Memory), line 10
depends on Section 9 (Structured AI Argumentation), line 12 depends on
Section 14 (the AI Patch Scheduler), and line 14 depends on Section 10
(Causal Mission Intelligence); all four are now implemented
(memory/emm.py's EpisodicMissionMemory,
agents/argumentation/aggregation.py's StructuredArgumentAggregator,
scheduler/algorithm6.py's AIPatchScheduler,
reasoning/causal/algorithm5.py's StructuralCausalEngine) -- pass
`memory=EpisodicMissionMemory()`,
`argumentation=StructuredArgumentAggregator(meta_memory=...)`,
`scheduler=AIPatchScheduler()`, and `causal_engine=StructuralCausalEngine(...)`
to enable them. Their `Null*` counterparts remain the default, explicit
opt-outs, wired via the same dependency-injection pattern throughout: with
no concrete implementation supplied, the Null* reference implementations
make each stage either return its honest empty result or raise
NotImplementedError, which this pipeline catches and records as a warning
on TTRNResult rather than fabricating a downstream value.

Line 12's condition ("if the reconciled decision requires patch
scheduling") is not inferred from `delta(m,t)`'s free-text claim -- doing
so would require fabricating a claim-classification heuristic the
manuscript does not specify. Instead the caller states the condition
directly by passing (or omitting) `patch_scheduling_request` to `infer()`.

Note on state coordinates: Section 4.1 treats the scheduler's output as
replacing the same delta(m,t) coordinate Section 9 produces ("delta(m,t)
<- AIPatchScheduler(delta(m,t), ...)"). This module keeps them as two
separate `TTRNResult` fields (`reconciled_decision` from Section 9,
`scheduling_result` from Section 14) instead, so that `agents/argumentation/`
does not need to depend on `scheduler/` -- a deliberate, documented
engineering deviation from the paper's single-coordinate notation, not a
change to either algorithm's actual logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.argumentation.interfaces import (
    ArgumentationReconcilerInterface,
    NullArgumentationReconciler,
    ReconciledDecision,
)
from agents.hypothesis import AgentHypothesis, HypothesisGeneratorInterface
from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.features import extract_feature_vector
from memory.interfaces import EpisodicMemoryInterface, MemoryRetrievalResult, NullEpisodicMemory
from memory.outcome import MissionOutcome
from reasoning.causal.interfaces import CausalContext, CausalEngineInterface, NullCausalEngine
from reasoning.state import CognitiveMissionState
from reasoning.trust.bayesian import BayesianTrustPosterior, bayesian_trust_update
from reasoning.trust.gnn import MissionAwareGNN, PredictionHeadOutputs
from scheduler.interfaces import (
    NullPatchScheduler,
    PatchSchedulerInterface,
    PatchSchedulingRequest,
    PatchSchedulingResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetryEdge:
    """One edge of newly arrived telemetry, ingested at Algorithm 1 line 1."""

    source: str
    target: str
    confidence: float
    risk: float = 0.0
    decay_rate: float = 0.05
    validity_window: float = float("inf")
    priority: float = 1.0
    dependency: tuple[str, ...] = ()


@dataclass(frozen=True)
class TTRNResult:
    """The tuple Algorithm 1 line 16 returns, plus the full CMI state
    sigma(m,t) and any warnings about stages that could not run."""

    state: CognitiveMissionState
    predictions: PredictionHeadOutputs
    posterior: BayesianTrustPosterior
    hypotheses: tuple[AgentHypothesis, ...]
    reconciled_decision: ReconciledDecision | None
    scheduling_result: PatchSchedulingResult | None
    causal_context: CausalContext | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


class TTRNPipeline:
    """Orchestrates Algorithm 1 over a TemporalReasoningMemoryGraph.

    Every collaborator is dependency-injected so the pipeline itself does
    not hardcode which concrete EMM / argumentation / causal implementation
    is in use -- required by this phase's honesty constraint, since those
    concrete implementations mostly do not exist yet, and required
    architecturally per the project's "dependency injection where
    appropriate" code-quality standard.
    """

    def __init__(
        self,
        graph: TemporalReasoningMemoryGraph,
        gnn: MissionAwareGNN,
        *,
        k_hop: int = 6,
        memory: EpisodicMemoryInterface | None = None,
        hypothesis_generators: tuple[HypothesisGeneratorInterface, ...] = (),
        argumentation: ArgumentationReconcilerInterface | None = None,
        causal_engine: CausalEngineInterface | None = None,
        scheduler: PatchSchedulerInterface | None = None,
        default_prior_variance: float = 1.0,
        observation_variance: float = 0.1,
    ) -> None:
        if not 1 <= k_hop <= 6:
            raise ValueError(f"k_hop must be in [1,6] per Section 5.1, got {k_hop}")
        self.graph = graph
        self.gnn = gnn
        self.k_hop = k_hop
        self.memory: EpisodicMemoryInterface = memory or NullEpisodicMemory()
        self.hypothesis_generators = hypothesis_generators
        self.argumentation: ArgumentationReconcilerInterface = (
            argumentation or NullArgumentationReconciler()
        )
        self.causal_engine: CausalEngineInterface = causal_engine or NullCausalEngine()
        self.scheduler: PatchSchedulerInterface = scheduler or NullPatchScheduler()
        self.default_prior_variance = default_prior_variance
        self.observation_variance = observation_variance

    def infer(
        self,
        mission_node: str,
        now: float,
        node_features: dict[str, np.ndarray],
        telemetry: tuple[TelemetryEdge, ...] = (),
        prior_posterior: BayesianTrustPosterior | None = None,
        outcome: MissionOutcome | None = None,
        outcome_confidence: float = 1.0,
        candidate_interventions: tuple[Any, ...] = (),
        patch_scheduling_request: PatchSchedulingRequest | None = None,
    ) -> TTRNResult:
        """TTRN-Infer(G_t, m, A): run Algorithm 1 for mission node `m` at
        time `now`, given newly arrived `telemetry` edges and the raw
        per-node feature vectors the mission-aware GNN needs (Section 7.1).

        `outcome` is Algorithm 2 line 3's `PendingOutcome(m)`: pass a
        `MissionOutcome` when this call is resolving a mission that has
        concluded (so EMM-Update appends an episode and, once a real memory
        backend is configured, becomes eligible for consolidation);
        omit it (the default) for an in-progress mission with no outcome
        yet to record.

        `candidate_interventions` is Algorithm 5's own input (Section
        10.5): a tuple of `reasoning.causal.algorithm5.CandidateIntervention`
        (typed `Any` here so this module does not need to import that
        concrete type, matching `CausalEngineInterface`'s own signature).
        Omit it (the default) to skip causal counterfactual generation for
        this query.

        `patch_scheduling_request` is Algorithm 1 line 11's condition,
        made explicit: pass a `PatchSchedulingRequest`
        (scheduler/interfaces.py) when this reconciled decision requires
        patch scheduling; omit it (the default) otherwise, in which case
        lines 12-13 are skipped without any warning (an ordinary,
        expected outcome, not a missing-component gap).
        """
        warnings: list[str] = []

        # Line 1: decay, then ingest telemetry.
        self.graph.apply_decay(now)
        for edge in telemetry:
            self.graph.ingest_telemetry_edge(
                edge.source,
                edge.target,
                timestamp=now,
                confidence=edge.confidence,
                risk=edge.risk,
                decay_rate=edge.decay_rate,
                validity_window=edge.validity_window,
                priority=edge.priority,
                dependency=edge.dependency,
            )

        # Line 2: k-hop neighborhood.
        neighborhood = self.graph.k_hop_neighborhood(mission_node, self.k_hop)

        # Line 3: EMM-Retrieve. `features` (Section 8, Algorithm 2 line 2 /
        # Algorithm 3 line 1) is extracted once and reused for both retrieve
        # and update below.
        features = extract_feature_vector(self.graph, neighborhood)
        retrieval: MemoryRetrievalResult = self.memory.retrieve(features, now)

        # Line 4-5: attention embedding + prediction heads.
        embedding = self.gnn.embed(self.graph, mission_node, node_features, now)
        predictions = self.gnn.predict(embedding)

        # Line 6: Bayesian update.
        prior = prior_posterior or BayesianTrustPosterior(
            mean=0.5, variance=self.default_prior_variance
        )
        posterior = bayesian_trust_update(prior, predictions.trust, self.observation_variance)

        node = self.graph.get_node(mission_node)
        node.history.append_trust(now, posterior.mean)
        node.history.append_uncertainty(now, posterior.variance)
        node.history.append_reasoning(
            now, f"trust_posterior={posterior.mean:.4f}", source="TTRNPipeline.infer"
        )

        # Lines 7-9: hypothesis generation (independent, executed sequentially here).
        hypotheses = tuple(
            gen.generate(neighborhood, posterior.mean, retrieval, domain_evidence=None)
            for gen in self.hypothesis_generators
        )

        # Line 10: structured argument aggregation.
        reconciled: ReconciledDecision | None
        try:
            reconciled = self.argumentation.aggregate(hypotheses)
        except NotImplementedError as exc:
            warnings.append(str(exc))
            reconciled = None

        # rho_agent(n) (Section 6.2): now that argumentation has (or hasn't)
        # run, log whether each hypothesis was accepted into Extension(AG).
        # Only recorded when a real reconciliation actually happened
        # (extension_hypotheses is populated) -- an escalated or
        # not-configured result carries no genuine acceptance information
        # to log.
        if reconciled is not None and reconciled.extension_hypotheses:
            accepted_ids = {h.agent_id for h in reconciled.extension_hypotheses}
            for hypothesis in hypotheses:
                node.history.append_agent(
                    now, hypothesis.agent_id, hypothesis.agent_id in accepted_ids
                )

        if reconciled is not None and reconciled.escalated:
            warnings.append(
                f"Structured AI Argumentation (Section 9.4) escalated to a human "
                f"operator for mission {mission_node} at t={now}: "
                f"{reconciled.summary}"
            )

        # Lines 11-13: patch scheduling (Section 14), only when the caller
        # has stated the condition of line 11 by supplying a request.
        scheduling_result: PatchSchedulingResult | None = None
        if patch_scheduling_request is not None:
            try:
                scheduling_result = self.scheduler.schedule(reconciled, patch_scheduling_request)
            except NotImplementedError as exc:
                warnings.append(str(exc))

        # Line 14: evidence assembly + causal counterfactual.
        causal_context: CausalContext | None
        try:
            causal_context = self.causal_engine.generate_counterfactual(
                reconciled, candidate_interventions=candidate_interventions
            )
        except NotImplementedError as exc:
            warnings.append(str(exc))
            causal_context = None

        # Line 15: EMM-Update.
        try:
            self.memory.update(
                mission_node,
                now,
                reconciled,
                posterior.mean,
                evidence=retrieval,
                causal_context=causal_context,
                features=features,
                agent_hypotheses=hypotheses,
                outcome=outcome,
                outcome_confidence=outcome_confidence,
                embedding=tuple(float(x) for x in embedding),
            )
        except NotImplementedError as exc:
            warnings.append(str(exc))

        state = CognitiveMissionState(
            mission_id=mission_node,
            time=now,
            embedding=tuple(float(x) for x in embedding),
            trust_posterior=posterior.mean,
            uncertainty=posterior.variance,
            retrieved_memory=retrieval.episodes,
            argument_graph=reconciled.argument_graph if reconciled is not None else None,
            reconciled_decision=reconciled,
            causal_context=causal_context,
        )

        logger.info(
            "TTRN-Infer(m=%s, t=%s): T_post=%.4f Upsilon=%.4f |N_k(m)|=%d warnings=%d",
            mission_node,
            now,
            posterior.mean,
            posterior.variance,
            len(neighborhood),
            len(warnings),
        )

        return TTRNResult(
            state=state,
            predictions=predictions,
            posterior=posterior,
            hypotheses=hypotheses,
            reconciled_decision=reconciled,
            scheduling_result=scheduling_result,
            causal_context=causal_context,
            warnings=tuple(warnings),
        )
