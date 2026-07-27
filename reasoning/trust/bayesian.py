"""Bayesian Trust Propagation / Posterior Update (manuscript Section 7.3).

A Bayesian trust-propagation layer sits between the raw GNN output (and the
deterministic Trust Propagation input T_in(v,t) of Section 7.2) and the
values consumed by Section 9's argumentation layer: rather than treating
T(m,t) as a point estimate, the layer maintains a posterior over T(m,t) that
is *updated* -- not recomputed from scratch -- as each new piece of TRMG or
memory evidence arrives, which is what allows the framework to report
calibrated uncertainty alongside every prediction (Algorithm 1, line 6:
"(T_post, Upsilon) <- BayesianUpdate(T, prior_{t-1})").

The manuscript specifies the *properties* this update must have (a genuine
posterior mean and a genuine, generally-decreasing uncertainty term -
Proposition 3, Section 5.4) without pinning down a specific likelihood
family. This module implements the standard, minimal instance of Bayesian
belief revision that satisfies those properties exactly: 1-D Gaussian
precision-weighted fusion of a prior estimate with a new observation
(the same update rule used by a scalar Kalman filter measurement step).
This is a deliberate, documented modeling choice -- not a claim that the
manuscript specifies a Kalman filter -- made so Proposition 3 can be
verified directly against a concrete implementation (see
tests/trust/test_bayesian.py) rather than left as an unimplemented promise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BayesianTrustPosterior:
    """(T_post(m,t), Upsilon(m,t)): posterior mean and uncertainty (variance)."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")


def bayesian_trust_update(
    prior: BayesianTrustPosterior,
    observation_mean: float,
    observation_variance: float,
) -> BayesianTrustPosterior:
    """(T_post, Upsilon) <- BayesianUpdate(T, prior_{t-1}) (Algorithm 1, line 6).

    Precision-weighted (inverse-variance) fusion of the prior posterior with
    a new observation (e.g. the GNN's trust head output, with an
    observation variance reflecting the GNN's own calibrated confidence):

        precision_post = 1/variance_prior + 1/variance_obs
        mean_post      = (mean_prior/variance_prior + observation_mean/variance_obs)
                          / precision_post
        variance_post  = 1 / precision_post

    Guarantees Proposition 3 (Section 5.4) by construction: whenever
    observation_variance is finite and positive, variance_post <
    variance_prior strictly (posterior uncertainty is non-increasing, and
    strictly decreasing, when genuine new evidence -- as opposed to merely
    applying temporal decay -- is incorporated).
    """
    if prior.variance <= 0:
        raise ValueError("prior.variance must be > 0 for precision-weighted fusion")
    if observation_variance <= 0:
        raise ValueError("observation_variance must be > 0 for precision-weighted fusion")

    prior_precision = 1.0 / prior.variance
    obs_precision = 1.0 / observation_variance
    posterior_precision = prior_precision + obs_precision

    posterior_mean = (
        prior.mean * prior_precision + observation_mean * obs_precision
    ) / posterior_precision
    posterior_variance = 1.0 / posterior_precision

    return BayesianTrustPosterior(mean=posterior_mean, variance=posterior_variance)
