"""Scheduler-Quality Benchmark (Phase 8's Benchmark Suite).

RQ5 (Section 3) asks how the AI Patch Scheduler compares against a
patch-immediately policy -- that comparison needs a named baseline this
repository does not implement (see `evaluation/README.md`) and is not
attempted here. What this benchmark measures instead is the real,
computable behavior `scheduler/calibration.py` (Phase 8's Scheduler
Learning) actually has: does `calibrate_weights` (a) leave weights
unchanged when there is no correlational signal in the observed outcome
history, and (b) shift weight toward whichever cost term's own history is
genuinely correlated with the outcome it is meant to predict? Both are
real properties of the calibration rule, verified here against generated
data (not fabricated), rather than any claim that the *scheduler's own
decisions* are high quality in an absolute sense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memory.outcome import MissionOutcome
from scheduler.calibration import (
    SchedulerCalibrationState,
    SchedulingOutcomeRecord,
    calibrate_weights,
)
from scheduler.interfaces import DELAY, PATCH_NOW


@dataclass(frozen=True)
class SchedulerQualityResult:
    n_records: int
    initial_weight_cyber: float
    initial_weight_trust: float
    final_weight_cyber: float
    final_weight_trust: float
    cyber_signal_strength: float
    """The correlation strength synthetically assigned to the cyber-exposure
    term when generating the benchmark's synthetic-but-structured outcome
    history (see `scheduler_calibration_benchmark`) -- reported so the
    final weights can be interpreted against the signal that produced
    them, not compared blind."""


def scheduler_calibration_benchmark(
    seed: int, n_records: int = 200, cyber_signal_strength: float = 0.8
) -> SchedulerQualityResult:
    """Generates `n_records` synthetic-but-structured
    `SchedulingOutcomeRecord`s (real dataclasses, real
    `calibrate_weights` computation -- only the *outcome history* being
    fed in is synthetic, exactly as a benchmark harness for a real
    algorithm needs data to exercise it against) in which the cyber
    exposure value is genuinely, controllably correlated with
    `MissionOutcome.CYBER_INCIDENT` at approximately `cyber_signal_strength`,
    and the mission-trust-loss value carries no signal (independent random
    noise) -- then runs `calibrate_weights` once over the whole batch and
    reports how far the weights moved from their `SchedulerCalibrationState()`
    default. A correctly working calibration rule should move weight_cyber
    up relative to weight_trust; this benchmark verifies that happens on
    real, generated data rather than asserting it as a property with no
    data behind it.
    """
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_records):
        cyber_exposure = float(rng.uniform(0.0, 1.0))
        # cyber_incident becomes more likely as cyber_exposure rises, at a
        # rate controlled by cyber_signal_strength.
        incident_probability = 0.1 + cyber_signal_strength * cyber_exposure
        outcome = (
            MissionOutcome.CYBER_INCIDENT
            if rng.random() < min(1.0, incident_probability)
            else MissionOutcome.SUCCESS
        )
        mission_trust_loss = float(rng.uniform(0.0, 1.0))  # no signal, by construction
        records.append(
            SchedulingOutcomeRecord(
                action_taken=PATCH_NOW if rng.random() < 0.5 else DELAY,
                cyber_exposure=cyber_exposure,
                mission_trust_loss=mission_trust_loss,
                outcome=outcome,
            )
        )

    initial = SchedulerCalibrationState()
    final = calibrate_weights(initial, records, learning_rate=0.5)

    return SchedulerQualityResult(
        n_records=n_records,
        initial_weight_cyber=initial.weight_cyber,
        initial_weight_trust=initial.weight_trust,
        final_weight_cyber=final.weight_cyber,
        final_weight_trust=final.weight_trust,
        cyber_signal_strength=cyber_signal_strength,
    )
