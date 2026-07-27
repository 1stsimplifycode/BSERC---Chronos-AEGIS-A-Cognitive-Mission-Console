"""Cybersecurity Contribution 2, Section 13: Temporal Attack Graphs.

"We retain, unchanged, the temporal attack graph formalism from our
preceding formulation: a conventional attack graph's node-exploit-
privilege sequence is extended with an explicit temporal and mission-phase
dimension, so that a path from aircraft to software to firmware to
communication to mission phase to time is scored not only on whether it
exists structurally but on whether the mission phase it would affect is
imminent. We note the connection to Section 12.2's AI-driven attack chain
reasoning explicitly: a temporal attack graph path is itself a candidate
feature vector s (Section 8.2.3) that can be matched against Episodic
Memory, so that a currently observed partial attack path can be compared
against how structurally similar past paths resolved, rather than scored
on structural existence alone."

INTEGRATION WITH THE TRMG (not a separate graph): an `AttackStep` is not a
node in some parallel attack-graph data structure -- it names a (source,
target) pair that must already be a real edge in the caller's
`TemporalReasoningMemoryGraph` (`validate_attack_path` enforces this).
Structural scoring reuses Section 7.2's exact `rho(u->v,t)` formula
(`reasoning.trust.propagation`) along the path's own edges; the Episodic
Memory match reuses Section 12.2's `AttackChainReasoning` unchanged
(`cyber.attack_chain`), via a `FeatureVector` built from the path's own
TRMG node attributes rather than a new feature schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from graph.trmg.model import TemporalReasoningMemoryGraph
from memory.features import FeatureVector
from reasoning.trust.propagation import propagated_trust_contribution


@dataclass(frozen=True)
class AttackStep:
    """One node-exploit-privilege hop (Section 13), backed by a real TRMG
    edge (source, target)."""

    source: str
    target: str
    exploit: str
    privilege_level: str


@dataclass(frozen=True)
class AttackPath:
    """An ordered sequence of `AttackStep`s: "a path from aircraft to
    software to firmware to communication to mission phase to time"
    (Section 13)."""

    steps: tuple[AttackStep, ...]

    @property
    def entry_node(self) -> str:
        if not self.steps:
            raise ValueError("an attack path must have at least one step")
        return self.steps[0].source

    @property
    def target_node(self) -> str:
        if not self.steps:
            raise ValueError("an attack path must have at least one step")
        return self.steps[-1].target


def validate_attack_path(graph: TemporalReasoningMemoryGraph, path: AttackPath) -> None:
    """Every step must be backed by a real TRMG edge, and consecutive
    steps must chain (step[i].target == step[i+1].source) -- the
    "integrate into the TRMG instead of existing separately" requirement
    enforced as a precondition, not a comment.
    """
    if not path.steps:
        raise ValueError("an attack path must have at least one step")
    for i, step in enumerate(path.steps):
        if not graph.has_edge(step.source, step.target):
            raise ValueError(
                f"attack step {i} ({step.source!r} -> {step.target!r}) has no backing TRMG edge"
            )
        if i > 0 and path.steps[i - 1].target != step.source:
            raise ValueError(
                f"attack path is not contiguous at step {i}: "
                f"{path.steps[i - 1].target!r} != {step.source!r}"
            )


def path_structural_score(
    graph: TemporalReasoningMemoryGraph, path: AttackPath, now: float
) -> float:
    """Whether the path "exists structurally" (Section 13): the product
    of rho(u->v,t) (Section 7.2) along the path's own specific edges --
    not the widest path between its endpoints (`cyber.risk_propagation`
    answers a different question, "what is the *best-attenuated* route,"
    whereas this is "how much does *this observed* route survive").
    """
    validate_attack_path(graph, path)
    score = 1.0
    for step in path.steps:
        edge = graph.get_edge(step.source, step.target)
        score *= propagated_trust_contribution(edge, now)
    return score


def mission_phase_imminence(now: float, phase_time: float | None, decay_rate: float = 0.1) -> float:
    """Whether "the mission phase [the path] would affect is imminent"
    (Section 13).

    ASSUMPTION (documented): the manuscript names imminence as a scoring
    factor without giving its closed form. This module models it as
    exponential decay in time-until-phase, mirroring the same decay
    family already used throughout this codebase (Section 6.1's edge
    decay; `memory.similarity.recency_weight`) rather than an unrelated
    scoring function invented for this module alone.

    `phase_time=None` (the affected mission phase has no known scheduled
    time) yields imminence 0.0 -- an unscheduled phase is honestly "not
    imminent," not defaulted to some fabricated middle value.
    """
    if phase_time is None:
        return 0.0
    time_until = max(0.0, phase_time - now)
    return math.exp(-decay_rate * time_until)


@dataclass(frozen=True)
class AttackPathScore:
    structural_score: float
    imminence: float
    temporal_score: float
    """structural_score * imminence: Section 13's combined "not only...
    but" scoring."""
    summary: str


def score_attack_path(
    graph: TemporalReasoningMemoryGraph,
    path: AttackPath,
    now: float,
    *,
    mission_phase_time: float | None,
    imminence_decay_rate: float = 0.1,
) -> AttackPathScore:
    """Section 13's combined score: structural existence times temporal
    imminence."""
    structural = path_structural_score(graph, path, now)
    imminence = mission_phase_imminence(now, mission_phase_time, imminence_decay_rate)
    temporal = structural * imminence
    return AttackPathScore(
        structural_score=structural,
        imminence=imminence,
        temporal_score=temporal,
        summary=(
            f"structural={structural:.4f} * imminence={imminence:.4f} "
            f"-> temporal_score={temporal:.4f}"
        ),
    )


def attack_path_feature_vector(
    graph: TemporalReasoningMemoryGraph,
    path: AttackPath,
    target_criticality_tier: str | None = None,
) -> FeatureVector:
    """ "[A] temporal attack graph path is itself a candidate feature
    vector s (Section 8.2.3) that can be matched against Episodic Memory"
    (Section 13) -- reuses Section 12.2's own attack-chain matching fields
    (`memory.features.FeatureVector`) rather than a parallel schema:

    - `entry_node_type` is read from the path's entry node's own TRMG
      attributes (the same field Section 12.2 matches "same entry node
      type" on).
    - `dependency_identity` is read from the first dependency-tagged node
      the path passes through (Section 12.2's "same intermediate
      dependency").
    - `target_criticality_tier` has no TRMG-derivable source the way the
      other two do (there is no established convention for a node to
      self-report its own criticality tier), so it is caller-supplied
      rather than guessed; `None` by default (honestly "not classified"),
      not a fabricated tier.

    Once built, match this feature vector against Episodic Memory via
    `cyber.attack_chain.AttackChainReasoning.predict` (Section 12.2) --
    this function only builds the query; it does not duplicate retrieval.
    """
    entry_node = graph.get_node(path.entry_node)
    entry_node_type = entry_node.attributes.get("entry_node_type")

    dependency_identity: str | None = None
    for step in path.steps:
        node = graph.get_node(step.target)
        candidate = node.attributes.get("dependency_identity")
        if candidate is not None:
            dependency_identity = str(candidate)
            break

    return FeatureVector(
        dependency_identity=dependency_identity,
        entry_node_type=str(entry_node_type) if entry_node_type is not None else None,
        target_criticality_tier=target_criticality_tier,
    )
