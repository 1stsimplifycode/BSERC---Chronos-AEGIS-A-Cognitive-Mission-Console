"""Trust Calibration (Phase 8's evidence-driven calibration requirement;
manuscript Section 17.5's named metrics: "Calibration curve" and "Trust
calibration error (ECE)").

Section 17.5, Table 3:
    Calibration curve         Reliability diagram of T_post vs. observed
                               outcome frequency, binned by predicted trust
    Trust calibration error   Sum_b (n_b/N) * |acc(b) - conf(b)| over bins
    (ECE)                     b of predicted T_post

This module implements both metrics exactly as named, plus a real,
evidence-driven recalibration mechanism built directly on top of them:
histogram-binning calibration (Zadrozny & Elkan, 2001) -- a standard,
well-known technique that maps a raw predicted trust value to the
*empirically observed* success frequency of the bin it falls in, rather
than any manually tuned constant. This is the same "realized-success
indicator" convention `memory/meta.py`'s Brier-score calibration already
uses (1.0 if `MissionOutcome.SUCCESS` else 0.0), applied here to
`BayesianTrustPosterior.mean` instead of agent confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory.outcome import MissionOutcome


def _realized_success(outcome: MissionOutcome) -> float:
    """The same binary realized-success indicator memory/meta.py's
    AgentCalibration uses: 1.0 if the mission succeeded, else 0.0."""
    return 1.0 if outcome == MissionOutcome.SUCCESS else 0.0


@dataclass(frozen=True)
class CalibrationBin:
    """One bin b of Table 3's reliability diagram / ECE sum."""

    bin_index: int
    lower: float
    upper: float
    count: int
    mean_confidence: float
    """conf(b): the mean predicted T_post of every prediction in this bin."""
    mean_accuracy: float
    """acc(b): the fraction of this bin's missions that actually succeeded."""


@dataclass(frozen=True)
class CalibrationReport:
    """Table 3's calibration curve (`bins`) and trust calibration error
    (`ece`), computed together from the same predictions/outcomes pass."""

    bins: tuple[CalibrationBin, ...]
    ece: float
    n: int


def compute_calibration(
    predictions: list[float] | tuple[float, ...],
    outcomes: list[MissionOutcome] | tuple[MissionOutcome, ...],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Section 17.5's calibration curve and ECE, computed over
    `predictions` (predicted T_post values, each in [0,1]) paired
    positionally with `outcomes` (the realized `MissionOutcome` for that
    same prediction).

    ECE = Sum_b (n_b/N) * |acc(b) - conf(b)|, over `n_bins` equal-width
    bins spanning [0,1] -- Table 3's own formula, applied literally.
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have the same length")
    if not predictions:
        raise ValueError("compute_calibration requires at least one prediction")
    if any(not 0.0 <= p <= 1.0 for p in predictions):
        raise ValueError("every prediction must be in [0,1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    n = len(predictions)
    edges = [i / n_bins for i in range(n_bins + 1)]
    bin_predictions: list[list[float]] = [[] for _ in range(n_bins)]
    bin_successes: list[list[float]] = [[] for _ in range(n_bins)]

    for pred, outcome in zip(predictions, outcomes, strict=True):
        # The top bin is closed on both ends ([1 - 1/n_bins, 1]); every
        # other bin is half-open ([lower, upper)) -- standard ECE binning.
        idx = min(int(pred * n_bins), n_bins - 1)
        bin_predictions[idx].append(pred)
        bin_successes[idx].append(_realized_success(outcome))

    bins = []
    ece = 0.0
    for i in range(n_bins):
        count = len(bin_predictions[i])
        mean_conf = sum(bin_predictions[i]) / count if count else 0.0
        mean_acc = sum(bin_successes[i]) / count if count else 0.0
        bins.append(
            CalibrationBin(
                bin_index=i,
                lower=edges[i],
                upper=edges[i + 1],
                count=count,
                mean_confidence=mean_conf,
                mean_accuracy=mean_acc,
            )
        )
        if count:
            ece += (count / n) * abs(mean_acc - mean_conf)

    return CalibrationReport(bins=tuple(bins), ece=ece, n=n)


class HistogramBinningCalibrator:
    """Evidence-driven recalibration (Phase 8's Trust Calibration
    requirement: "driven by observed evidence, not manually tuned
    constants"): maps a raw predicted trust value to the empirically
    observed success frequency of the bin it falls in (histogram-binning
    calibration, Zadrozny & Elkan 2001).

    A bin with zero observations has no evidence to recalibrate against;
    `calibrate` returns the raw prediction unchanged for such a bin rather
    than fabricating a correction.
    """

    def __init__(self, report: CalibrationReport) -> None:
        self._report = report

    @property
    def report(self) -> CalibrationReport:
        return self._report

    def calibrate(self, raw_prediction: float) -> float:
        if not 0.0 <= raw_prediction <= 1.0:
            raise ValueError(f"raw_prediction must be in [0,1], got {raw_prediction}")
        n_bins = len(self._report.bins)
        idx = min(int(raw_prediction * n_bins), n_bins - 1)
        b = self._report.bins[idx]
        return b.mean_accuracy if b.count > 0 else raw_prediction
