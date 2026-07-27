"""Statistical analysis (Section 17.6) for the Evaluation Framework.

Descriptive statistics, confidence intervals, Cohen's d, and the
Holm-Bonferroni correction use only the standard library. Real paired
significance tests (paired t-test, Wilcoxon signed-rank) are computed only
when `scipy` is installed -- an optional dependency this repository does
not otherwise use (see `paired_comparison`'s docstring) -- so that no
p-value is ever fabricated from a hand-rolled approximation of the t- or
Wilcoxon-signed-rank distribution.
"""

from __future__ import annotations

import math
import statistics as stats
from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptiveStats:
    """n, mean, sample standard deviation, and a confidence interval."""

    n: int
    mean: float
    stdev: float
    ci_low: float
    ci_high: float
    confidence: float


def descriptive_stats(values: tuple[float, ...], confidence: float = 0.95) -> DescriptiveStats:
    """Mean, sample standard deviation, and a confidence interval for
    `values`.

    ASSUMPTION (documented): the interval uses the standard normal
    quantile (`statistics.NormalDist`) rather than the Student's
    t-distribution, since the standard library has no t-distribution
    quantile function and this repository does not depend on scipy. This
    is exact as n grows large and understates interval width somewhat for
    small n -- a documented simplification, not a fabricated exact value.
    With a single observation the interval collapses to a point at the
    mean (stdev is undefined for n=1, reported as 0.0).
    """
    if len(values) == 0:
        raise ValueError("descriptive_stats requires at least one value")
    n = len(values)
    mean = stats.fmean(values)
    if n == 1:
        return DescriptiveStats(
            n=1, mean=mean, stdev=0.0, ci_low=mean, ci_high=mean, confidence=confidence
        )
    sd = stats.stdev(values)
    z = stats.NormalDist().inv_cdf(0.5 + confidence / 2)
    margin = z * sd / math.sqrt(n)
    return DescriptiveStats(
        n=n, mean=mean, stdev=sd, ci_low=mean - margin, ci_high=mean + margin, confidence=confidence
    )


def cohens_d(sample_a: tuple[float, ...], sample_b: tuple[float, ...]) -> float:
    """Cohen's d for two independent samples, using the pooled standard
    deviation. Returns 0.0 (no detectable effect) when the pooled standard
    deviation itself is 0.0, rather than dividing by zero."""
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        raise ValueError("cohens_d requires at least 2 observations per sample")
    mean_a, mean_b = stats.fmean(sample_a), stats.fmean(sample_b)
    var_a, var_b = stats.variance(sample_a), stats.variance(sample_b)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


def holm_bonferroni(p_values: tuple[float, ...], alpha: float = 0.05) -> tuple[bool, ...]:
    """Holm-Bonferroni step-down correction: for each p-value in
    `p_values` (original order), returns whether it is rejected at
    family-wise error rate `alpha`. Standard library only -- this
    procedure needs only sorting and arithmetic, no special functions.
    """
    m = len(p_values)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break  # Holm's procedure stops at the first non-rejection.
    return tuple(rejected)


@dataclass(frozen=True)
class PairedComparisonResult:
    """A paired comparison between two equal-length samples (Section
    17.6)."""

    n: int
    mean_difference: float
    t_statistic: float | None
    p_value_t: float | None
    p_value_wilcoxon: float | None
    cohens_d_paired: float
    note: str


def paired_comparison(
    sample_a: tuple[float, ...], sample_b: tuple[float, ...]
) -> PairedComparisonResult:
    """Paired comparison between two equal-length samples (e.g. RQ13's
    superadditivity test: full architecture vs. one component removed,
    same scenario/seed pairs).

    The mean difference, t-statistic, and Cohen's d are always computed
    from the standard library. The two named p-values (paired t-test,
    Wilcoxon signed-rank) require `scipy`, an optional dependency not
    otherwise used anywhere in this repository: when scipy is not
    installed, both p-value fields are `None` and `note` explains why --
    an honest "not computed," matching this project's `Null*`/
    `NotImplementedError` convention for a genuinely unavailable
    component, rather than approximating either distribution by hand.
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("paired_comparison requires equal-length samples")
    n = len(sample_a)
    if n < 2:
        raise ValueError("paired_comparison requires at least 2 paired observations")

    differences = tuple(a - b for a, b in zip(sample_a, sample_b, strict=True))
    mean_diff = stats.fmean(differences)
    sd_diff = stats.stdev(differences)
    t_statistic = None if sd_diff == 0.0 else mean_diff / (sd_diff / math.sqrt(n))
    d = cohens_d(sample_a, sample_b)

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return PairedComparisonResult(
            n=n,
            mean_difference=mean_diff,
            t_statistic=t_statistic,
            p_value_t=None,
            p_value_wilcoxon=None,
            cohens_d_paired=d,
            note=(
                "scipy is not installed: p-values were not computed (the "
                "t-statistic and Cohen's d above are exact stdlib "
                "computations; see module docstring)"
            ),
        )

    t_res = scipy_stats.ttest_rel(sample_a, sample_b)
    try:
        w_res = scipy_stats.wilcoxon(sample_a, sample_b)
        p_wilcoxon: float | None = float(w_res.pvalue)
    except ValueError:
        # e.g. every paired difference is exactly zero -- wilcoxon() itself
        # raises rather than returning a value; report None, not a fabricated 1.0.
        p_wilcoxon = None

    return PairedComparisonResult(
        n=n,
        mean_difference=mean_diff,
        t_statistic=float(t_res.statistic),
        p_value_t=float(t_res.pvalue),
        p_value_wilcoxon=p_wilcoxon,
        cohens_d_paired=d,
        note="t-statistic and p-values computed via scipy.stats.ttest_rel / wilcoxon",
    )
