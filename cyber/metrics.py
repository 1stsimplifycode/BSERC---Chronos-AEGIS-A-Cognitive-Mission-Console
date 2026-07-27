"""Section 12.6: New Mission-Aware Cyber Metrics.

"We formally define three metrics intended to replace or supplement
static, per-component CVSS scoring with mission-aware quantities computed
over the TRMG." All three "are formal definitions requiring empirical
calibration and validation against the digital twin's ground truth
(Section 15); RQ12 (Section 3) is designed to test whether they correlate
more strongly with injected ground-truth mission consequence than CVSS
scores alone, and we do not claim that correlation here" -- neither does
this module: it computes the metrics exactly as defined, and reports
their intermediate terms so a caller (and the Explainability Pipeline,
Section 10.7) can inspect and eventually calibrate them, rather than
asserting they are accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cyber.centrality import (
    betweenness_centrality,
    dependency_nodes,
    dependency_subgraph_adjacency,
)
from cyber.risk_propagation import propagate_risk_attenuation_with_decay
from graph.trmg.model import TemporalReasoningMemoryGraph

# -- 12.6.1: Mission Trust Exposure Score (MTES) -----------------------------


@dataclass(frozen=True)
class MTESResult:
    """MTES(v) = integral_t^{t+w} rho(v) * attenuation(v->mission, tau) *
    psi(mission_criticality) d(tau)  (Section 12.6.1)."""

    value: float
    intrinsic_risk: float
    attenuation_at_t: float
    aggregate_decay_rate: float
    mission_criticality_weight: float
    window: float
    summary: str


def mission_trust_exposure_score(
    graph: TemporalReasoningMemoryGraph,
    vulnerability_node: str,
    mission_node: str,
    now: float,
    window: float,
    *,
    intrinsic_risk: float,
    mission_criticality: float,
    k_hop: int = 6,
) -> MTESResult:
    """MTES(v) (Section 12.6.1): rho(v) is `intrinsic_risk` (caller-supplied
    -- this repository has no vulnerability database / SBOM ingestion to
    derive it from, see cyber/README.md); attenuation(v->mission) is
    computed by Section 12.1's widest-path propagation
    (cyber.risk_propagation); psi(mission_criticality) is this module's
    documented choice psi(kappa_m) = kappa_m (identity), since the
    manuscript names psi as a weighting function without giving its closed
    form and kappa_m in/ [0,1] is already Section 7.4's established scale
    for mission criticality.

    CLOSED-FORM DERIVATION (documented assumption): attenuation(v->mission,
    tau) decays forward in time as attenuation(v->mission,t) *
    exp(-Lambda*(tau-t)), where Lambda is the sum of decay rates along the
    winning (widest) path at time t (`propagate_risk_attenuation_with_decay`).
    This assumes the same path remains the widest path throughout the
    integration window [t, t+w] -- the manuscript does not address whether
    the maximal-attenuation path can change mid-window, and recomputing it
    at every instant is not performed here. Under that assumption the
    integral has the exact closed form

        integral_0^w exp(-Lambda*s) ds = (1 - exp(-Lambda*w)) / Lambda   (Lambda > 0)
                                        = w                               (Lambda = 0)

    so MTES(v) = rho(v) * attenuation(v->mission,t) * psi(kappa_m) * [that factor].
    """
    if intrinsic_risk < 0.0:
        raise ValueError(f"intrinsic_risk must be >= 0, got {intrinsic_risk}")
    if not 0.0 <= mission_criticality <= 1.0:
        raise ValueError(f"mission_criticality must be in [0,1], got {mission_criticality}")
    if window < 0.0:
        raise ValueError(f"window must be >= 0, got {window}")

    path_info = propagate_risk_attenuation_with_decay(graph, vulnerability_node, now, k_hop)
    attenuation, decay_rate = path_info.get(mission_node, (0.0, 0.0))

    psi = mission_criticality  # documented identity choice, see docstring

    if window == 0.0 or attenuation == 0.0:
        time_integral_factor = 0.0
    elif decay_rate <= 0.0:
        time_integral_factor = window
    else:
        time_integral_factor = (1.0 - math.exp(-decay_rate * window)) / decay_rate

    value = intrinsic_risk * attenuation * psi * time_integral_factor

    return MTESResult(
        value=value,
        intrinsic_risk=intrinsic_risk,
        attenuation_at_t=attenuation,
        aggregate_decay_rate=decay_rate,
        mission_criticality_weight=psi,
        window=window,
        summary=(
            f"MTES({vulnerability_node})={value:.6f} = rho={intrinsic_risk:.4f} * "
            f"attenuation={attenuation:.4f} * psi={psi:.4f} * "
            f"time_integral_factor={time_integral_factor:.4f} over window={window}"
        ),
    )


# -- 12.6.2: Mission Dependency Risk Index (MDRI) ----------------------------


@dataclass(frozen=True)
class MDRIResult:
    """MDRI(m) = sum_{v in Dep(m)} rho(v) * centrality(v, TRMG) (Section 12.6.2)."""

    value: float
    dependency_nodes: frozenset[str]
    centrality: dict[str, float]
    contributions: dict[str, float]
    summary: str


def mission_dependency_risk_index(
    graph: TemporalReasoningMemoryGraph,
    mission_node: str,
    *,
    intrinsic_risk: dict[str, float],
    k_hop: int = 6,
) -> MDRIResult:
    """MDRI(m) (Section 12.6.2). `intrinsic_risk` maps dependency node id
    to rho(v); a dependency node absent from `intrinsic_risk` is treated
    as rho(v)=0.0 -- "no known vulnerability risk recorded for this
    dependency," the honest default, not an error, since this repository
    has no vulnerability database to look rho(v) up in (see cyber/README.md).
    Centrality is betweenness centrality restricted to dependency edges
    (Section 12.6.2's own named example; see cyber/centrality.py).
    """
    dep_nodes = dependency_nodes(graph, mission_node, k_hop)
    adjacency = dependency_subgraph_adjacency(graph, dep_nodes)
    centrality = betweenness_centrality(adjacency)

    contributions = {v: intrinsic_risk.get(v, 0.0) * centrality.get(v, 0.0) for v in dep_nodes}
    value = sum(contributions.values())

    return MDRIResult(
        value=value,
        dependency_nodes=dep_nodes,
        centrality=centrality,
        contributions=contributions,
        summary=f"MDRI({mission_node})={value:.6f} over {len(dep_nodes)} dependency node(s)",
    )


# -- 12.6.3: Cyber Mission Criticality Score (CMCS) --------------------------


@dataclass(frozen=True)
class CMCSResult:
    """CMCS(v,m) = MTES(v) * [1 + MDRI-normalized centrality of v within
    Dep(m)] * P_success(m,t)^-1 (Section 12.6.3)."""

    value: float
    mtes: MTESResult
    mdri: MDRIResult
    normalized_centrality: float
    p_success: float
    summary: str


def cyber_mission_criticality_score(
    graph: TemporalReasoningMemoryGraph,
    vulnerability_node: str,
    mission_node: str,
    now: float,
    window: float,
    *,
    intrinsic_risk: dict[str, float],
    mission_criticality: float,
    p_success: float,
    k_hop: int = 6,
) -> CMCSResult:
    """CMCS(v,m) (Section 12.6.3), combining MTES (12.6.1), MDRI (12.6.2),
    and P_success(m,t) (Section 7's mission-aware GNN output).

    ASSUMPTION (documented): "MDRI-normalized centrality of v within
    Dep(m)" is not spelled out numerically in the manuscript. This module
    reads it as v's own contribution to MDRI -- rho(v)*centrality(v,TRMG)
    -- expressed as a fraction of the mission's total MDRI(m): v's
    proportional share of the mission's total dependency risk index,
    which is the natural reading of "centrality-weighted risk normalized
    by MDRI" and reduces to 0 for a vulnerability node that is not itself
    one of Dep(m)'s dependency nodes (e.g. a network-entry vulnerability
    rather than a supply-chain one), which is the honest value in that
    case, not an error.
    """
    if p_success <= 0.0:
        raise ValueError(
            f"p_success must be > 0 to invert (Section 12.6.3's P_success(m,t)^-1), got {p_success}"
        )

    mtes = mission_trust_exposure_score(
        graph,
        vulnerability_node,
        mission_node,
        now,
        window,
        intrinsic_risk=intrinsic_risk.get(vulnerability_node, 0.0),
        mission_criticality=mission_criticality,
        k_hop=k_hop,
    )
    mdri = mission_dependency_risk_index(
        graph, mission_node, intrinsic_risk=intrinsic_risk, k_hop=k_hop
    )

    own_contribution = mdri.contributions.get(vulnerability_node, 0.0)
    normalized_centrality = own_contribution / mdri.value if mdri.value != 0.0 else 0.0

    value = mtes.value * (1.0 + normalized_centrality) * (1.0 / p_success)

    return CMCSResult(
        value=value,
        mtes=mtes,
        mdri=mdri,
        normalized_centrality=normalized_centrality,
        p_success=p_success,
        summary=(
            f"CMCS({vulnerability_node},{mission_node})={value:.6f} = "
            f"MTES={mtes.value:.6f} * [1 + {normalized_centrality:.4f}] * "
            f"(1/P_success={1.0 / p_success:.4f})"
        ),
    )
