"""The nine domain-specialized agents (manuscript Section 9.1, retained).

"We retain, unchanged, the nine domain-specialized agents from our
preceding TTRN-only formulation: Aircraft Engineer, Cyber Analyst,
Satellite Planner, Weather Analyst, Mission Planner, Maintenance Engineer,
Communication Specialist, Supply Chain Analyst, and Safety Officer. Each
generates an independent hypothesis about the current mission decision
from its own slice of the TRMG, the mission-aware GNN's outputs, and
retrieved Episodic Memory relevant to its domain."

This module enumerates the nine agent identities Section 5.3's complexity
analysis counts as `|A| = 9`. It does NOT implement any of their concrete
domain reasoning: the manuscript names each agent's role but does not
specify the reasoning logic that turns "its own slice of the TRMG" into a
claim, so inventing that logic here would be exactly the kind of
paper-unfaithful fabrication this repository's design philosophy forbids.
`agents/hypothesis.py`'s `HypothesisGeneratorInterface` is the contract a
future concrete implementation of each of these nine roles must satisfy.
"""

from __future__ import annotations

AIRCRAFT_ENGINEER = "aircraft_engineer"
CYBER_ANALYST = "cyber_analyst"
SATELLITE_PLANNER = "satellite_planner"
WEATHER_ANALYST = "weather_analyst"
MISSION_PLANNER = "mission_planner"
MAINTENANCE_ENGINEER = "maintenance_engineer"
COMMUNICATION_SPECIALIST = "communication_specialist"
SUPPLY_CHAIN_ANALYST = "supply_chain_analyst"
SAFETY_OFFICER = "safety_officer"

DOMAIN_AGENT_ROSTER: tuple[str, ...] = (
    AIRCRAFT_ENGINEER,
    CYBER_ANALYST,
    SATELLITE_PLANNER,
    WEATHER_ANALYST,
    MISSION_PLANNER,
    MAINTENANCE_ENGINEER,
    COMMUNICATION_SPECIALIST,
    SUPPLY_CHAIN_ANALYST,
    SAFETY_OFFICER,
)
"""|A| = 9 (Section 5.3): the fixed agent identity set every complexity
analysis and every Section 9 formula (Conflict(H), credibility weighting)
is stated over."""
