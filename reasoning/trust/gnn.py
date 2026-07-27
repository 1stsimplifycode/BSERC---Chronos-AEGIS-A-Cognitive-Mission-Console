"""The Mission-Aware Graph Neural Network (manuscript Section 7.1).

Rather than performing node classification or link prediction as an end in
itself, the mission-aware GNN is trained to produce six coupled outputs for
a given mission node m at time t, from a shared learned embedding h_m(t):
mission trust T(m,t), mission success probability P_success(m,t), expected
mission delay D(m,t), cyber impact C(m,t), communication reliability
R_comm(m,t), and operational confidence O(m,t). The embedding h_m(t) is
produced by a temporal graph-attention mechanism operating over the TRMG's
k-hop neighborhood of m together with retrieved memory M_ret, attending
preferentially to edges and memory entries with high confidence-adjusted-
for-decay and high priority (i.e. the same rho(u->v,t) quantity Section 7.2
defines for Trust Propagation).

IMPLEMENTATION STATUS -- read before using this module's numbers anywhere:
This module implements the *architecture* exactly as specified above: real
attention weights (softmax over rho(u->v,t) for each neighbor of m), a real
weighted aggregation into an embedding, and real linear prediction heads
mapping that embedding to the six outputs. What it does NOT do is claim to
be *trained*: the six heads' weight matrices are randomly initialized (via
an explicit, seedable RNG) because no training data or dataset exists yet
-- Section 17's evaluation protocol, which would fit these weights, has
"not yet been executed" per the manuscript's own abstract. Any numbers this
module produces are therefore an untrained model's outputs: useful for
exercising the pipeline's data flow and for testing the attention mechanism
itself, but not a claim about mission trust, success probability, or any
other prediction quality. Do not report these outputs as evaluation
results.

PHASE 8: the six heads' weights (but not `_feature_projection`, the
embedding layer) are now mutable via `get_head_weights`/`set_head_weights`,
so `reasoning.continual.algorithm7.MissionAdapt` (Section 16.5) can apply
the regularized parameter update of Section 16.2 between missions. This
does not change the "untrained at construction" fact above -- a freshly
constructed `MissionAwareGNN` is exactly as untrained as before; only a
caller that explicitly runs Algorithm 7 changes that, and each such
update is itself documented at the point it happens, never silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from graph.trmg.model import TemporalReasoningMemoryGraph
from reasoning.trust.propagation import propagated_trust_contribution


@dataclass(frozen=True)
class PredictionHeadOutputs:
    """The six coupled outputs of Section 7.1: (T, P_success, D, C, R_comm, O)."""

    trust: float
    p_success: float
    expected_delay: float
    cyber_impact: float
    comm_reliability: float
    operational_confidence: float


class MissionAwareGNN:
    """Attention-weighted embedding + six linear prediction heads over a
    TRMG neighborhood, per Section 7.1.

    Parameters
    ----------
    embedding_dim:
        Dimensionality d of h_m(t).
    feature_dim:
        Dimensionality of the raw per-node feature vectors the caller
        supplies (there is no synthetic default -- see `embed`).
    seed:
        RNG seed for the (untrained) prediction-head weight matrices, so
        results are reproducible across runs without claiming to be a
        fitted model.
    """

    def __init__(self, embedding_dim: int = 16, feature_dim: int = 8, seed: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        rng = np.random.default_rng(seed)
        # Projects a raw feature vector into embedding space before attention
        # pooling. Untrained (see module docstring).
        self._feature_projection = rng.normal(
            scale=1.0 / math.sqrt(feature_dim), size=(feature_dim, embedding_dim)
        )
        # Six untrained linear heads, embedding_dim -> scalar.
        self._heads = {
            name: rng.normal(scale=1.0 / math.sqrt(embedding_dim), size=embedding_dim)
            for name in (
                "trust",
                "p_success",
                "expected_delay",
                "cyber_impact",
                "comm_reliability",
                "operational_confidence",
            )
        }

    def embed(
        self,
        graph: TemporalReasoningMemoryGraph,
        mission_node: str,
        node_features: dict[str, np.ndarray],
        now: float,
        retrieved_memory_features: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """h_m(t): temporal graph-attention embedding over N_k(m) union M_ret.

        `node_features` must supply a real feature vector (shape
        `(feature_dim,)`) for `mission_node` and for every one of its
        in-neighbors that should participate in attention; nodes without a
        supplied feature vector are simply excluded from the weighted sum
        rather than defaulted to a fabricated vector. Raises ValueError if
        `mission_node` itself has no supplied feature vector, since an
        embedding with no self-representation would silently be pure
        neighbor-only signal.

        Attention weight for neighbor u of v=mission_node is
        rho(u->v,t) (Section 7.2), softmax-normalized across all
        participating neighbors plus retrieved-memory feature vectors
        (each memory feature vector is attended to with weight 1.0 prior to
        softmax, since Episodic Memory's own confidence/decay scoring,
        Section 8.4, is not yet implemented -- see memory/interfaces.py).
        """
        if mission_node not in node_features:
            raise ValueError(
                f"node_features must include a feature vector for mission_node={mission_node!r}"
            )

        incoming = graph.in_edges(mission_node)
        neighbor_ids = [u for u in incoming if u in node_features]
        weights: list[float] = [
            propagated_trust_contribution(incoming[u], now) for u in neighbor_ids
        ]
        vectors: list[np.ndarray] = [node_features[u] for u in neighbor_ids]

        memory_vectors = retrieved_memory_features or []
        weights.extend([1.0] * len(memory_vectors))
        vectors.extend(memory_vectors)

        self_vector = node_features[mission_node]
        projected_self = self_vector @ self._feature_projection

        if not vectors:
            # No attended neighbors or memory: embedding is purely self-derived.
            return projected_self

        projected = np.stack([v @ self._feature_projection for v in vectors])
        w = np.asarray(weights, dtype=float)
        # Softmax normalization (numerically stabilized).
        w = np.exp(w - w.max())
        w = w / w.sum()
        aggregated = (projected * w[:, None]).sum(axis=0)

        # Mission node's own (decay-free) representation always contributes,
        # combined with the attention-aggregated neighborhood at equal weight.
        return 0.5 * projected_self + 0.5 * aggregated

    def get_head_weights(self) -> dict[str, np.ndarray]:
        """theta: a copy of the six prediction heads' weight vectors
        (Section 7.1). A copy, not a view, so a caller cannot mutate this
        GNN's live parameters by mutating the returned dict/arrays --
        `set_head_weights` is the only sanctioned mutation path (Phase 8,
        Section 16.2/16.5's regularized parameter update, Algorithm 7 line
        4, needs a public way to read and write theta; reaching into the
        private `_heads` attribute from outside this class was the
        alternative this method replaces).
        """
        return {name: w.copy() for name, w in self._heads.items()}

    def set_head_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Replace theta with `weights` (Algorithm 7 line 4's theta update).
        `weights` must supply every one of the six head names with an
        array of shape `(embedding_dim,)`; raises ValueError otherwise
        rather than silently applying a partial or malformed update.
        """
        expected_names = set(self._heads)
        if set(weights) != expected_names:
            raise ValueError(
                f"set_head_weights requires exactly {sorted(expected_names)}, "
                f"got {sorted(weights)}"
            )
        for name, w in weights.items():
            if w.shape != (self.embedding_dim,):
                raise ValueError(
                    f"head {name!r} must have shape ({self.embedding_dim},), got {w.shape}"
                )
        self._heads = {name: w.copy() for name, w in weights.items()}

    def predict(self, embedding: np.ndarray) -> PredictionHeadOutputs:
        """(T, P_success, D, C, R_comm, O) <- PredictionHeads(h_m(t))
        (Algorithm 1, line 5). Outputs are squashed to their natural ranges
        ([0,1] for trust/probability/confidence/impact; >=0 for delay) via a
        sigmoid or softplus, since the raw dot product of an untrained head
        is otherwise unbounded.
        """
        raw = {name: float(embedding @ w) for name, w in self._heads.items()}

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        def softplus(x: float) -> float:
            return math.log1p(math.exp(-abs(x))) + max(x, 0.0)

        return PredictionHeadOutputs(
            trust=sigmoid(raw["trust"]),
            p_success=sigmoid(raw["p_success"]),
            expected_delay=softplus(raw["expected_delay"]),
            cyber_impact=sigmoid(raw["cyber_impact"]),
            comm_reliability=sigmoid(raw["comm_reliability"]),
            operational_confidence=sigmoid(raw["operational_confidence"]),
        )
