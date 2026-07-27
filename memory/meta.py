"""Meta Memory M_m (manuscript Section 8.2.4) and its recompute step
(Algorithm 2, line 9).

"M_m holds knowledge about the framework's own reasoning quality rather
than about missions directly: per-agent confidence calibration curves,
historical reasoning-quality scores, agent reliability estimates (Section
9.3's credibility weights, tracked over time)..."

Algorithm 2, line 9: "Update M_m: recompute per-agent calibration and
credibility from the outcome o vs. each agent's prior hypothesis in A_hyp."
The word "recompute" (not "increment" or "update incrementally") is taken
literally here: `MetaMemory.recompute` rebuilds calibration state from a
full pass over the given entries every time it is called, which is also
what makes it safe to call repeatedly from batched consolidation
(memory/consolidation.py) without double-counting episodes across runs.

Calibration metric: Section 9.1 defines an AgentHypothesis as (claim,
evidence, confidence, counterarguments) -- there is no structured
"predicted outcome" field to grade against MissionOutcome's five
categories. Absent that structure, this module uses the only quantity
actually available -- confidence -- against a binary realized-success
indicator (1.0 if MissionOutcome.SUCCESS else 0.0), i.e. the per-hypothesis
squared error term of a Brier score. This is a documented simplification
of "reasoning-quality scores" (Section 8.2.4), not a claim that Section 9's
full credibility semantics are implemented; see agents/argumentation/ for
what is genuinely missing there.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from memory.episodic import EpisodicEntry
from memory.outcome import MissionOutcome


@dataclass
class AgentCalibration:
    """Per-agent calibration curve summary (Section 8.2.4)."""

    agent_id: str
    num_observations: int = 0
    mean_brier_error: float = 0.0

    @property
    def credibility(self) -> float:
        """Section 9.3's credibility weight, as tracked over time by M_m.
        1.0 minus mean Brier error, floored at 0.0."""
        return max(0.0, 1.0 - self.mean_brier_error)


class MetaMemory:
    """M_m: per-agent calibration and credibility, recomputed from Episodic
    Memory (Algorithm 2, line 9)."""

    def __init__(self) -> None:
        self._calibration: dict[str, AgentCalibration] = {}

    def recompute(self, entries: Iterable[EpisodicEntry]) -> None:
        """Rebuild calibration state from scratch over `entries`."""
        calibration: dict[str, AgentCalibration] = {}
        for entry in entries:
            realized = 1.0 if entry.outcome == MissionOutcome.SUCCESS else 0.0
            for hypothesis in entry.agent_hypotheses:
                calib = calibration.setdefault(
                    hypothesis.agent_id, AgentCalibration(agent_id=hypothesis.agent_id)
                )
                error = (hypothesis.confidence - realized) ** 2
                n = calib.num_observations
                calib.mean_brier_error = (calib.mean_brier_error * n + error) / (n + 1)
                calib.num_observations = n + 1
        self._calibration = calibration

    def credibility(self, agent_id: str) -> float | None:
        calib = self._calibration.get(agent_id)
        return calib.credibility if calib else None

    def all_calibrations(self) -> dict[str, AgentCalibration]:
        return dict(self._calibration)
