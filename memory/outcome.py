"""Mission outcome categories (manuscript Section 8.2.3).

An episodic entry's `o` field is "the eventually observed mission outcome
(success, failure, delay, cyber incident, or another outcome category
defined by the mission digital twin's ground truth, Section 15)."

The Mission Digital Twin (Section 15) is not yet implemented (see
digital_twin/README.md), so nothing in this codebase can look an outcome up
automatically; `MissionOutcome` is always supplied explicitly by a caller
who has ground truth (e.g. a human-in-the-loop mission debrief, or,
eventually, the digital twin), never inferred.
"""

from __future__ import annotations

from enum import StrEnum


class MissionOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    DELAY = "delay"
    CYBER_INCIDENT = "cyber_incident"
    OTHER = "other"
