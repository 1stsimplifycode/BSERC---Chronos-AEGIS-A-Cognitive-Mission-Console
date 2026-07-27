"""The Explainability Pipeline: Eight Levels of Explanation (manuscript
Section 10.7).

"We consolidate, as a named Explainability Pipeline, the eight distinct
levels at which CMI is able to answer an operator's question of why, none
of which is a substitute for the others": System-level, Mission-level,
Agent-level, Graph-level, Evidence-level, Counterfactual-level,
Trust-level, and Scheduler-level. "We require that a full explanation
response be assemblable by traversing exactly these eight levels in a
fixed order, rather than by an ad hoc combination decided per query, so
that two structurally similar decisions produce structurally comparable
explanations."

This module assembles those eight levels from the real, already-computed
outputs of the other CMI components a caller supplies via
`ExplanationInputs` -- it does not compute anything new itself, and it
does not fabricate content for a level whose underlying component was not
exercised for this query: Scheduler-level (Section 14.1) is reported
unavailable whenever `scheduling_result` is not supplied (e.g. patch
scheduling was not requested for this particular decision), and populated
for real -- action, feasibility, and cost terms -- when it is.

Cyber reasoning (Sections 11-13) does not get a ninth level -- Section
10.7 fixes exactly eight -- it is woven into the existing ones it
naturally belongs to: Cognitive Cyber Reasoning's retrieved episodes
(Section 11's mu(q)) surface under Evidence-level alongside Section 9.7's
own retrieved memory; risk-propagation attenuation and the MDRI
dependency chain (Sections 12.1, 12.6.2) surface under Graph-level
alongside the TRMG neighborhood and dependency volatility they are
computed from. Both are additive: supplying `cyber_consequence`,
`cyber_exposure`, and/or `dependency_risk` enriches those levels' content
without changing their fixed position or the other six levels' behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.hypothesis import AgentHypothesis

SYSTEM_LEVEL_STATEMENT = (
    "CMI produces this class of recommendation because it couples temporal-graph "
    "reasoning (TTRN), persistent memory (EMM), structured multi-party reasoning "
    "(argumentation), and causal explanation into one state-transition system "
    "sigma(m,t) = Phi(sigma(m,t-1), e_t) (Section 4.1), evaluated across the four "
    "coupled dimensions of temporal evolution, structural dependency, uncertainty, "
    "and mission context (Section 5.1) at every query -- not by four independently "
    "computed subsystems combined only at the end."
)


@dataclass(frozen=True)
class ExplanationLevel:
    name: str
    content: str
    available: bool = True


@dataclass(frozen=True)
class ExplanationReport:
    """Exactly eight levels, always in the fixed order Section 10.7 requires."""

    levels: tuple[ExplanationLevel, ...]

    def level(self, name: str) -> ExplanationLevel:
        for lvl in self.levels:
            if lvl.name == name:
                return lvl
        raise KeyError(name)


@dataclass(frozen=True)
class ExplanationInputs:
    """Everything the eight levels are assembled from -- all of it real
    output already produced by other CMI components; this class supplies
    no defaults that would silently stand in for missing data.
    """

    mission_id: str
    trust_posterior: float
    uncertainty: float
    neighborhood_size: int
    hypotheses: tuple[AgentHypothesis, ...] = ()
    credibility_weights: dict[str, float] = field(default_factory=dict)
    extension_agent_ids: frozenset[str] | None = None
    dependency_volatility: float | None = None
    trust_floor: float | None = None
    mission_criticality: float | None = None
    retrieved_episodes: tuple[Any, ...] = ()
    semantic_rules: tuple[Any, ...] = ()
    causal_results: tuple[Any, ...] = ()
    scheduling_result: Any | None = None
    """A `scheduler.interfaces.PatchSchedulingResult`, typed `Any` here so
    this module does not need to depend on the `scheduler` package."""
    cyber_consequence: Any | None = None
    """A `cyber.consequence.ConsequencePrediction` (Section 11's mu(q)),
    typed `Any` so this module does not need to depend on `cyber`."""
    cyber_exposure: Any | None = None
    """A `cyber.scheduler_integration.CyberExposureBreakdown` (Section
    12.6.1's MTES, as consumed by Algorithm 6), typed `Any` for the same
    reason."""
    dependency_risk: Any | None = None
    """A `cyber.metrics.MDRIResult` (Section 12.6.2), typed `Any` for the
    same reason."""


def assemble_explanation(inputs: ExplanationInputs) -> ExplanationReport:
    """Assemble the eight-level explanation in Section 10.7's fixed order."""
    levels = (
        _system_level(),
        _mission_level(inputs),
        _agent_level(inputs),
        _graph_level(inputs),
        _evidence_level(inputs),
        _counterfactual_level(inputs),
        _trust_level(inputs),
        _scheduler_level(inputs),
    )
    return ExplanationReport(levels=levels)


def _system_level() -> ExplanationLevel:
    return ExplanationLevel(name="system", content=SYSTEM_LEVEL_STATEMENT)


def _mission_level(inputs: ExplanationInputs) -> ExplanationLevel:
    if inputs.trust_floor is None or inputs.mission_criticality is None:
        return ExplanationLevel(
            name="mission",
            content="Mission Criticality Model (Section 7.4) inputs not supplied for this query.",
            available=False,
        )
    meets = "meets" if inputs.trust_posterior >= inputs.trust_floor else "does not meet"
    content = (
        f"mission {inputs.mission_id}'s criticality kappa_m={inputs.mission_criticality:.3f} "
        f"sets its trust floor tau(m)={inputs.trust_floor:.3f}; observed "
        f"T_post={inputs.trust_posterior:.3f} {meets} that floor."
    )
    return ExplanationLevel(name="mission", content=content)


def _agent_level(inputs: ExplanationInputs) -> ExplanationLevel:
    if not inputs.hypotheses:
        return ExplanationLevel(
            name="agent",
            content="no agent hypotheses were generated for this query.",
            available=False,
        )
    lines = []
    for h in inputs.hypotheses:
        credibility = inputs.credibility_weights.get(h.agent_id)
        accepted = (
            h.agent_id in inputs.extension_agent_ids
            if inputs.extension_agent_ids is not None
            else None
        )
        lines.append(
            f"{h.agent_id}: claim='{h.claim}' conf={h.confidence:.3f} "
            f"credibility={'n/a' if credibility is None else f'{credibility:.3f}'} "
            f"accepted={'n/a' if accepted is None else accepted}"
        )
    return ExplanationLevel(name="agent", content="; ".join(lines))


def _graph_level(inputs: ExplanationInputs) -> ExplanationLevel:
    volatility = (
        "n/a" if inputs.dependency_volatility is None else f"{inputs.dependency_volatility:.3f}"
    )
    parts = [
        f"decision drawn from the {inputs.neighborhood_size}-node k-hop neighborhood N_k(m) "
        f"(Section 5.1); dependency volatility Vol(n,t,w) (Section 6.5) = {volatility}."
    ]
    if inputs.cyber_exposure is not None:
        delay_mtes = inputs.cyber_exposure.delay
        parts.append(
            f"Risk propagation (Section 12.1): attenuation(v->mission)="
            f"{delay_mtes.attenuation_at_t:.4f}, aggregate path decay rate="
            f"{delay_mtes.aggregate_decay_rate:.4f}."
        )
    if inputs.dependency_risk is not None:
        parts.append(
            f"Dependency chain (Section 12.6.2): MDRI={inputs.dependency_risk.value:.4f} over "
            f"{len(inputs.dependency_risk.dependency_nodes)} dependency node(s)."
        )
    return ExplanationLevel(name="graph", content=" ".join(parts))


def _evidence_level(inputs: ExplanationInputs) -> ExplanationLevel:
    parts = []
    if inputs.retrieved_episodes or inputs.semantic_rules:
        parts.append(
            f"{len(inputs.retrieved_episodes)} retrieved episode(s) and "
            f"{len(inputs.semantic_rules)} semantic rule(s) cited as evidence (Section 9.7)."
        )
    if inputs.cyber_consequence is not None and inputs.cyber_consequence.mu is not None:
        parts.append(
            f"Cognitive Cyber Reasoning (Section 11): mu(q)={inputs.cyber_consequence.mu:.4f} "
            f"from {len(inputs.cyber_consequence.supporting_episodes)} retrieved "
            f"cyber-relevant episode(s)."
        )
    if not parts:
        return ExplanationLevel(
            name="evidence",
            content=(
                "no retrieved episodic memory, semantic rules, or cyber-consequence "
                "evidence were cited (Section 9.7's E(m,t))."
            ),
            available=False,
        )
    return ExplanationLevel(name="evidence", content=" ".join(parts))


def _counterfactual_level(inputs: ExplanationInputs) -> ExplanationLevel:
    if not inputs.causal_results:
        return ExplanationLevel(
            name="counterfactual",
            content="no candidate interventions were evaluated (Algorithm 5, Section 10.5).",
            available=False,
        )
    lines = [
        f"do({r.intervention.variable}={r.intervention.value}): "
        f"delta_T={r.delta_trust:+.4f} [{r.causal_status}]"
        for r in inputs.causal_results
    ]
    return ExplanationLevel(name="counterfactual", content="; ".join(lines))


def _trust_level(inputs: ExplanationInputs) -> ExplanationLevel:
    content = (
        f"T_post={inputs.trust_posterior:.4f}, Upsilon={inputs.uncertainty:.4f} "
        f"(Section 7.2 Trust Propagation feeding Section 7.3's Bayesian posterior update)."
    )
    return ExplanationLevel(name="trust", content=content)


def _scheduler_level(inputs: ExplanationInputs) -> ExplanationLevel:
    if inputs.scheduling_result is None:
        return ExplanationLevel(
            name="scheduler",
            content=(
                "no patch-scheduling decision was requested for this query "
                "(Algorithm 1, lines 11-13; Section 14.1)."
            ),
            available=False,
        )
    r = inputs.scheduling_result
    content = (
        f"action={r.action} selected (feasible={r.feasible}); cost terms: {r.cost} "
        f"(Algorithm 6, Section 14.1). {r.summary}"
    )
    return ExplanationLevel(name="scheduler", content=content)
