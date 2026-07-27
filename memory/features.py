"""ExtractFeatureVector (manuscript Section 8, Algorithm 2 line 2 / Algorithm
3 line 1): the shared feature extractor both memory update and memory
retrieval use to summarize a mission's TRMG-neighborhood state.

Section 8.2.3 describes the feature vector `s` as covering "vulnerable
dependency identity, communication topology class, weather category,
mission-phase identity, and other salient TRMG attributes." This module
reads those four categorical fields, plus any other numeric attributes,
directly from the `attributes` dict already carried by every `TRMGNode`
(graph/trmg/model.py) -- there is no separate schema to keep in sync.

Design choice, documented here rather than left implicit: if more than one
node in the neighborhood sets the same categorical key, this extractor
takes whichever value it encounters first while iterating
`neighborhood_node_ids` (a `set`, so iteration order is unspecified and not
guaranteed stable across runs) rather than attempting to resolve a conflict
the manuscript does not specify how to resolve. Callers who need
conflict-free extraction should ensure only one node in the neighborhood
(e.g. the mission node itself, or a single upstream weather/topology node)
sets a given key.

Section 12.2 (Cybersecurity Contribution 1, AI-Driven Attack Chain
Reasoning) retrieves episodic memory "in the same manner as Section 11's
Cognitive Cyber Reasoning but specifically restricted to attack-chain-
shaped episodes," matched by "same entry node type, same intermediate
dependency, same target criticality tier." `dependency_identity` above
already serves as "intermediate dependency"; `entry_node_type` and
`target_criticality_tier` are added here as two more categorical fields
of the same `s` (Section 8.2.3's "other salient TRMG attributes"), read
from the same node `attributes` dict, so Section 12.2 reuses Algorithm 3
exactly rather than needing a parallel feature schema (see cyber/attack_chain.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graph.trmg.model import TemporalReasoningMemoryGraph

_CATEGORICAL_KEYS = (
    "dependency_identity",
    "topology_class",
    "weather_category",
    "mission_phase",
    "entry_node_type",
    "target_criticality_tier",
)


@dataclass(frozen=True)
class FeatureVector:
    """s: a feature vector summarizing a mission's TRMG-neighborhood state
    (Section 8.2.3), extended with Section 12.2's attack-chain matching
    fields."""

    dependency_identity: str | None = None
    topology_class: str | None = None
    weather_category: str | None = None
    mission_phase: str | None = None
    entry_node_type: str | None = None
    target_criticality_tier: str | None = None
    continuous_features: dict[str, float] = field(default_factory=dict)


def extract_feature_vector(
    graph: TemporalReasoningMemoryGraph, neighborhood_node_ids: set[str]
) -> FeatureVector:
    """ExtractFeatureVector(N_k(m)): scan every node in the given
    neighborhood for the four categorical keys, any other numeric node
    attributes, and (Phase 8) incoming-edge confidence, per this module's
    docstring.

    PHASE 8 addition: incoming-edge confidence -- chi(e).c (Section 6.1)
    for every edge whose target is in the neighborhood -- is folded into
    `continuous_features` as `"{source}->{target}.confidence"`. This closes
    a real retrieval-precision gap diagnosed while investigating RQ6/RQ7
    (`scripts/experiment_rq6_rq7_memory_retrieval.py`): most Digital Twin
    scenarios (`digital_twin/scenarios.py`) never set any of the six
    categorical keys on any node they touch, so `continuous_features` was
    the *only* other place `s` (Section 8.2.3) could carry a
    scenario-specific signal, and until this change nothing populated it
    either -- every episode's feature vector was identical regardless of
    scenario. Edge confidence, by contrast, is set by nearly every
    intervention event (`CommunicationLoss`, `SensorFailure`,
    `WeatherChange`, `SoftwareCompromise` all call
    `ingest_telemetry_edge`/create an edge with a scenario-specific
    confidence value) and is unambiguously "a salient TRMG attribute"
    (Section 8.2.3's own phrase) -- chi(e).c is literally what every other
    CMI component already treats as the graph's primary signal.
    """
    categorical: dict[str, str | None] = dict.fromkeys(_CATEGORICAL_KEYS)
    continuous: dict[str, float] = {}

    for node_id in neighborhood_node_ids:
        node = graph.get_node(node_id)
        for key, value in node.attributes.items():
            if key in _CATEGORICAL_KEYS:
                if categorical[key] is None and value is not None:
                    categorical[key] = str(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                continuous[f"{node_id}.{key}"] = float(value)

        for source, edge in graph.in_edges(node_id).items():
            continuous[f"{source}->{node_id}.confidence"] = edge.confidence

    return FeatureVector(
        dependency_identity=categorical["dependency_identity"],
        topology_class=categorical["topology_class"],
        weather_category=categorical["weather_category"],
        mission_phase=categorical["mission_phase"],
        entry_node_type=categorical["entry_node_type"],
        target_criticality_tier=categorical["target_criticality_tier"],
        continuous_features=continuous,
    )
